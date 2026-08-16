##############################################################################
#####                                                                    #####
#####                                                                    #####
#####     MuDCANet — PROPOSED MODEL (MudRing-tuned DenseNet + ECA)       #####
#####                                                                    #####
#####   Architecture : DenseNet backbone with interleaved ECA channel    #####
#####                  attention. Dense block → transition → ECA →      #####
#####                  MaxPool, repeated for 3 stages.                   #####
#####   Tuner        : Mud Ring Algorithm (MRA)                          #####
#####                  Desuky et al., IEEE Access, Vol.10, 2022.         #####
#####                                                                    #####
#####   Flow:                                                            #####
#####     - MudRing searches ONCE on fold 1 of CICIDS2017 binary.        #####
#####     - Best config transferred to all 4 variants:                   #####
#####         CICIDS2017 binary + multiclass                             #####
#####         UNSW-NB15  binary + multiclass                             #####
#####     - Each variant: 5-fold CV + final training + test evaluation.  #####
#####                                                                    #####
#####   All data comes from ids_pipeline (precomputed folds + final).    #####
##############################################################################

import os
import json
import time
import warnings
import math
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.metrics import f1_score

import ids_pipeline as ip
import baselines.ablation_base as ab

warnings.filterwarnings("ignore")


##############################################################################
# ────────────────────────────  CONFIGURATION  ─────────────────────────────
##############################################################################

MODEL_NAME = "Proposed"
OUTPUT_DIR = os.path.join(ip.BASE, "experiment_results", "MuDCANet")

# ─── MudRing settings ──────────────────────────────────────────────────────
MRA_PARTICLES  = 8
MRA_ITERATIONS = 5

# ─── 4-D HP search space (matches other tuner scripts) ─────────────────────
HP_SPACE = {
    "conv_dropout":  (0.1, 0.4),
    "dense_dropout": (0.2, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "l2_reg":        (1e-5, 1e-3),
}

# ─── DenseNet architectural constants ──────────────────────────────────────
GROWTH_RATE            = 16
DENSE_LAYERS_PER_BLOCK = 3

# ─── ECA constants (Wang et al., CVPR 2020, Eq. 1) ─────────────────────────
ECA_GAMMA = 2
ECA_B     = 1

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    se_ratio=8,   # accepted but unused (ECA has no reduction ratio)
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)

VARIANT_PLAN = [
    ("CICIDS2017", "binary"),
    ("CICIDS2017", "multiclass"),
    ("UNSW_NB15",  "binary"),
    ("UNSW_NB15",  "multiclass"),
]

ab.setup_seeds()
ab.setup_output_dirs(OUTPUT_DIR)
print(f"[MuDCANet] Output root: {OUTPUT_DIR}")


##############################################################################
# ─────────────────────────  MODEL DEFINITION  ─────────────────────────────
##############################################################################

def _eca_kernel_size(channels):
    """Adaptive 1D conv kernel size (Wang et al., CVPR 2020, Eq. 1)."""
    t = int(abs(math.log2(channels) / ECA_GAMMA + ECA_B / ECA_GAMMA))
    k = t if t % 2 == 1 else t + 1
    return max(1, k)


def _eca_block(x, name):
    """Efficient Channel Attention. Single 1D conv over GAP descriptor."""
    C = x.shape[-1]
    k = _eca_kernel_size(C)
    gap  = layers.GlobalAveragePooling2D(name=f"{name}_gap")(x)
    gap  = layers.Reshape((C, 1), name=f"{name}_rs1")(gap)
    attn = layers.Conv1D(1, kernel_size=k, padding="same",
                          use_bias=False, name=f"{name}_conv1d")(gap)
    attn = layers.Activation("sigmoid", name=f"{name}_sig")(attn)
    attn = layers.Reshape((1, 1, C), name=f"{name}_rs2")(attn)
    return layers.Multiply(name=f"{name}_scale")([x, attn])


def _dense_block(x, n_layers, growth_rate, l2_reg, name):
    """Dense block (Huang et al., CVPR 2017) with bottleneck 1x1 → 3x3."""
    reg = regularizers.l2(l2_reg)
    all_outputs = [x]
    for i in range(n_layers):
        lname = f"{name}_L{i+1}"
        fmap = (layers.Concatenate(name=f"{lname}_cat")(all_outputs)
                if len(all_outputs) > 1 else all_outputs[0])
        fmap = layers.BatchNormalization(name=f"{lname}_bn1")(fmap)
        fmap = layers.ReLU(name=f"{lname}_r1")(fmap)
        fmap = layers.Conv2D(4 * growth_rate, (1, 1), padding="same",
                             kernel_regularizer=reg, name=f"{lname}_bot")(fmap)
        fmap = layers.BatchNormalization(name=f"{lname}_bn2")(fmap)
        fmap = layers.ReLU(name=f"{lname}_r2")(fmap)
        fmap = layers.Conv2D(growth_rate, (3, 3), padding="same",
                             kernel_regularizer=reg, name=f"{lname}_c")(fmap)
        all_outputs.append(fmap)
    return layers.Concatenate(name=f"{name}_out")(all_outputs)


def _transition_layer(x, out_filters, l2_reg, dropout, name):
    """BN → ReLU → 1x1 Conv compression with optional SpatialDropout."""
    reg = regularizers.l2(l2_reg)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.ReLU(name=f"{name}_r")(x)
    x = layers.Conv2D(out_filters, (1, 1), padding="same",
                      kernel_regularizer=reg, name=f"{name}_c")(x)
    if dropout > 0:
        x = layers.SpatialDropout2D(dropout, name=f"{name}_sd")(x)
    return x


def build_proposed(input_shape, n_classes, task,
                   filters_1=64, filters_2=128, filters_3=256,
                   se_ratio=8,
                   dense_units_1=256, dense_units_2=128,
                   conv_dropout=0.25, dense_dropout=0.4,
                   learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    """DenseNet backbone with interleaved ECA attention."""
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(filters_1, (3, 3), padding="same",
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name="init_conv")(inputs)
    x = layers.BatchNormalization(name="init_bn")(x)
    x = layers.ReLU(name="init_r")(x)

    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, "db1")
    x = _transition_layer(x, filters_1, l2_reg, conv_dropout, "tr1")
    x = _eca_block(x, "eca1")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2

    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, "db2")
    x = _transition_layer(x, filters_2, l2_reg, conv_dropout, "tr2")
    x = _eca_block(x, "eca2")
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2

    x = _dense_block(x, DENSE_LAYERS_PER_BLOCK, GROWTH_RATE, l2_reg, "db3")
    x = _transition_layer(x, filters_3, l2_reg, 0.0, "tr3")
    x = _eca_block(x, "eca3")
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(dense_units_1,
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense1")(x)
    x = layers.BatchNormalization(name="bn_d1")(x)
    x = layers.ReLU(name="relu_d1")(x)
    x = layers.Dropout(dense_dropout, name="drop_d1")(x)

    x = layers.Dense(dense_units_2,
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name="dense2")(x)
    x = layers.BatchNormalization(name="bn_d2")(x)
    x = layers.ReLU(name="relu_d2")(x)
    x = layers.Dropout(dense_dropout, name="drop_d2")(x)

    outputs = (layers.Dense(1, activation="sigmoid", name="output")(x)
               if task == "binary" else
               layers.Dense(n_classes, activation="softmax", name="output")(x))
    model = keras.Model(inputs, outputs, name="Proposed_DenseECA")
    opt   = keras.optimizers.Adam(learning_rate=learning_rate)
    loss  = "binary_crossentropy" if task == "binary" else "categorical_crossentropy"
    model.compile(optimizer=opt, loss=loss,
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


def _full_cfg(tuned):
    c = dict(DEFAULT_HP)
    c.update(tuned)
    return c


##############################################################################
# ─────────────────────────  MUDRING SEARCH  ───────────────────────────────
##############################################################################

def _decode_position(position):
    keys = list(HP_SPACE.keys())
    cfg  = {}
    for i, key in enumerate(keys):
        val    = float(np.clip(position[i], 0.0, 1.0))
        lo, hi = HP_SPACE[key]
        if key in ("learning_rate", "l2_reg"):
            cfg[key] = float(10 ** (np.log10(lo) + val *
                                    (np.log10(hi) - np.log10(lo))))
        else:
            cfg[key] = float(lo + val * (hi - lo))
    return cfg


def _hp_objective(cfg, fold_data, task, n_classes):
    """Single MudRing HP trial. Returns validation F1."""
    tf.keras.backend.clear_session()
    feature_shape = fold_data["feature_shape"]
    model = build_proposed(feature_shape, n_classes, task, **_full_cfg(cfg))
    cw    = ip.get_class_weights(fold_data["y_int_train"], n_classes)
    model.fit(
        fold_data["X_train"], fold_data["y_train"],
        validation_data=(fold_data["X_val"], fold_data["y_val"]),
        epochs=20, batch_size=ab.BATCH_SIZE, class_weight=cw,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0)
    y_raw     = model.predict(fold_data["X_val"], verbose=0)
    y_pred, _ = ip.decode_predictions(y_raw, task)
    y_true    = fold_data["y_int_val"]
    return f1_score(y_true, y_pred,
                    average="binary" if task == "binary" else "macro",
                    zero_division=0)


def run_mudring_search(fold_data, task, n_classes):
    """Mud Ring Algorithm (Desuky et al., 2022).
    Equations:
      Eq.(2): a = 2*(1 - t/T_max)
      Eq.(1): K = 2*a*r - a
      Eq.(6): C = 2*r
      Eq.(3): D_new = D_rand - K*|C*D_rand - D_i|   [EXPLORE, |K|>=1]
      Eq.(4): A = |C*D* - D_i|
      Eq.(5): D_new = D**sin(2πl) - K*A             [EXPLOIT, |K|<1]
    """
    rng = np.random.default_rng(ab.RANDOM_STATE)
    dim = len(HP_SPACE)

    positions = rng.uniform(0.0, 1.0, (MRA_PARTICLES, dim))
    print(f"\n  MudRing: initialising {MRA_PARTICLES} dolphins ({dim}D)...")
    fitness = np.full(MRA_PARTICLES, -np.inf)
    for p in range(MRA_PARTICLES):
        fitness[p] = _hp_objective(_decode_position(positions[p]),
                                   fold_data, task, n_classes)

    best_idx  = int(np.argmax(fitness))
    gbest_pos = positions[best_idx].copy()
    gbest_val = float(fitness[best_idx])
    pbest_val = fitness.copy()
    iter_scores = []
    all_scores  = list(fitness)

    print(f"  Initial best F1: {gbest_val:.4f}")
    print(f"  Running {MRA_ITERATIONS} iterations...")

    for t in range(1, MRA_ITERATIONS + 1):
        a = 2.0 * (1.0 - t / MRA_ITERATIONS)   # Eq.(2)
        for p in range(MRA_PARTICLES):
            r  = rng.uniform(0.0, 1.0, dim)
            K  = (2.0 * a * r) - a              # Eq.(1)
            r2 = rng.uniform(0.0, 1.0, dim)
            C  = 2.0 * r2                       # Eq.(6)

            explore_mask = (np.abs(K) >= 1.0)
            exploit_mask = ~explore_mask
            new_pos = positions[p].copy()

            if explore_mask.any():
                j = rng.integers(0, MRA_PARTICLES)
                D_rand = positions[j]
                dist = np.abs(C * D_rand - positions[p])
                new_pos[explore_mask] = (
                    D_rand[explore_mask]
                    - K[explore_mask] * dist[explore_mask])   # Eq.(3)

            if exploit_mask.any():
                l = rng.uniform(-1.0, 1.0, dim)
                A = np.abs(C * gbest_pos - positions[p])      # Eq.(4)
                new_pos[exploit_mask] = (
                    gbest_pos[exploit_mask]
                    * np.sin(2.0 * np.pi * l[exploit_mask])
                    - K[exploit_mask] * A[exploit_mask])       # Eq.(5)

            positions[p] = np.clip(new_pos, 0.0, 1.0)
            score = _hp_objective(_decode_position(positions[p]),
                                  fold_data, task, n_classes)
            all_scores.append(score)
            if score > pbest_val[p]:
                pbest_val[p] = score
            if score > gbest_val:
                gbest_val = score
                gbest_pos = positions[p].copy()

        iter_scores.append(gbest_val)
        if t % max(1, MRA_ITERATIONS // 4) == 0 or t == 1:
            print(f"    Iter {t:3d}/{MRA_ITERATIONS}  "
                  f"a={a:.3f}  best F1={gbest_val:.4f}")

    best_cfg = _decode_position(gbest_pos)
    print(f"\n  MudRing complete. Best F1: {gbest_val:.4f}")
    print(f"  Best config: {best_cfg}")
    return best_cfg, iter_scores, all_scores, pbest_val


##############################################################################
# ───────────────────────  SEARCH-SPECIFIC SAVE  ──────────────────────────
##############################################################################

def save_search_arrays(iter_scores, all_scores, pbest_val, task, ds_tag):
    npy_dir = os.path.join(OUTPUT_DIR, "npy", task)
    tag     = f"{MODEL_NAME}_{task}_{ds_tag}"
    np.save(os.path.join(npy_dir, f"{tag}_mra_iter_scores.npy"),
            np.array(iter_scores))
    np.save(os.path.join(npy_dir, f"{tag}_mra_all_scores.npy"),
            np.array(all_scores))
    np.save(os.path.join(npy_dir, f"{tag}_mra_pbest.npy"),
            np.array(pbest_val))
    print(f"  Search arrays saved → npy/{task}/{tag}_mra_*.npy")


def save_best_hps(best_cfg, task, ds_tag):
    tag  = f"{MODEL_NAME}_{task}_{ds_tag}"
    path = os.path.join(OUTPUT_DIR, "npy", task, f"{tag}_best_hps.json")
    with open(path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"  Best HPs saved → npy/{task}/{tag}_best_hps.json")


##############################################################################
# ────────────────────────  VARIANT RUNNER  ────────────────────────────────
##############################################################################

def run_variant(dataset_name, task, best_full_cfg, best_tuned_cfg,
                all_summaries, save_search=False, search_data=None):
    """Train MuDCANet on one (dataset, task) with the transferred config."""
    ds_tag = ip.DATASET_TAG[dataset_name]
    print(f"\n{'='*65}")
    print(f"  {dataset_name} | {task}")
    if not save_search:
        print(f"  [Config transferred from CICIDS2017 binary search]")
    print(f"{'='*65}")
    print(f"\n{'─'*50}")
    print(f"  Using config: {best_tuned_cfg}")
    print(f"{'─'*50}")

    prefix = f"{MODEL_NAME}_{task}_{ds_tag}"
    t0     = time.time()

    (fold_m, test_m, final_h, cm,
     model, y_pred, y_prob, final) = ab.run_cv_and_final(
        build_proposed, best_full_cfg, dataset_name, task, OUTPUT_DIR, prefix)

    print(f"  Done in {time.time()-t0:.1f}s")

    ab.save_arrays(final["y_int_test"], y_pred, y_prob,
                   final["class_names"], MODEL_NAME, task, ds_tag, OUTPUT_DIR)
    ab.save_history(final_h, MODEL_NAME, task, ds_tag, OUTPUT_DIR)
    ab.save_cv_metrics(fold_m, MODEL_NAME, task, ds_tag, OUTPUT_DIR)
    ab.save_metrics_table(test_m, MODEL_NAME, task, ds_tag, OUTPUT_DIR)

    if save_search and search_data is not None:
        iter_sc, all_sc, pbest = search_data
        save_search_arrays(iter_sc, all_sc, pbest, task, ds_tag)
        save_best_hps(best_tuned_cfg, task, ds_tag)

    if task == "multiclass":
        ab.save_confusion_matrix(cm, final["class_names"],
                                 MODEL_NAME, task, ds_tag, OUTPUT_DIR)
        ab.save_roc_curves(final["y_int_test"], y_prob, final["class_names"],
                           MODEL_NAME, task, ds_tag, OUTPUT_DIR)

    summary = {
        "Dataset": dataset_name, "Task": task,
        "Proposed_Acc": test_m["Accuracy"],
        "Proposed_F1":  test_m["F1"],
        "Proposed_MCC": test_m["MCC"],
        "Proposed_AUC": test_m["AUC_ROC"],
        **{f"HP_{k}": v for k, v in best_tuned_cfg.items()},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, f"{prefix}_summary.csv"), index=False)
    all_summaries.append(summary)
    print(f"  Summary saved. DONE: {dataset_name} | {task}")
    tf.keras.backend.clear_session()


##############################################################################
# ────────────────────────  MAIN EXECUTION  ────────────────────────────────
##############################################################################

if __name__ == "__main__":
    all_summaries = []

    # ─── STEP 1: MudRing search on CICIDS2017 binary (fold 1) ───────────
    print(f"\n{'='*65}")
    print(f"  CICIDS2017 | binary  [MudRing Search]")
    print(f"{'='*65}")

    search_fold = ip.load_fold("CICIDS2017", "binary", 1)
    n_classes_src = search_fold["n_classes"]

    print(f"\n{'─'*50}")
    print(f"  MudRing Search (CICIDS2017 binary — fold 1)")
    print(f"{'─'*50}")
    t0 = time.time()
    best_tuned_cfg, iter_sc, all_sc, pbest = run_mudring_search(
        search_fold, "binary", n_classes_src)
    print(f"  Search done in {time.time()-t0:.1f}s")

    best_full_cfg = _full_cfg(best_tuned_cfg)
    search_data   = (iter_sc, all_sc, pbest)
    print(f"\n  Best config will be transferred to all remaining variants.")
    print(f"  Tuned HPs: {best_tuned_cfg}")

    # ─── STEPS 2-5: Train each variant with the transferred config ──────
    for i, (ds, task) in enumerate(VARIANT_PLAN):
        # Save search artifacts only on the source variant (CICIDS binary).
        is_source = (ds == "CICIDS2017" and task == "binary")
        run_variant(ds, task, best_full_cfg, best_tuned_cfg, all_summaries,
                    save_search=is_source,
                    search_data=search_data if is_source else None)

    # ─── Master summary ────────────────────────────────────────────────
    master = pd.DataFrame(all_summaries)
    master.to_csv(
        os.path.join(OUTPUT_DIR, "MuDCANet_master_summary.csv"), index=False)
    print(f"\nMaster summary saved.")
    print(master.to_string(index=False))
    print("\nMuDCANet COMPLETE.")