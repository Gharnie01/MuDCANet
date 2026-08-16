##############################################################################
# tuner_base.py
#
# Shared base for the three binary-only tuner scripts:
#   scripts3.py — HHO   (mealpy OriginalHHO)
#   scripts4.py — PSO   (PySwarms GlobalBestPSO)
#   scripts5.py — GWO   (mealpy OriginalGWO)
#
# Contains:
#   - Base CNN backbone (build_base_cnn)
#   - Shared HP search space + defaults
#   - decode_position / full_cfg / hp_objective
#   - Search-once-transfer flow: search runs on the source dataset (fold 1
#     of CICIDS2017 binary), best config transferred to UNSW-NB15 binary.
#   - Tuner-specific save helpers (search arrays, best HPs)
#   - CV + final training via ablation_base.run_cv_and_final
##############################################################################

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import ids_pipeline as ip
import baselines.ablation_base as ab


##############################################################################
# ────────────────────────────  CONSTANTS  ─────────────────────────────────
##############################################################################

BATCH_SIZE   = ab.BATCH_SIZE
MAX_EPOCHS   = ab.MAX_EPOCHS
RANDOM_STATE = ab.RANDOM_STATE

# Both tuners restricted to binary — the search transfer only makes sense
# task-wise between the same task on different datasets.
EXPERIMENT_PLAN = ["CICIDS2017", "UNSW_NB15"]

HP_SPACE = {
    "conv_dropout":  (0.1, 0.4),
    "dense_dropout": (0.2, 0.5),
    "learning_rate": (1e-4, 1e-2),
    "l2_reg":        (1e-5, 1e-3),
}

DEFAULT_HP = dict(
    filters_1=64, filters_2=128, filters_3=256,
    dense_units_1=256, dense_units_2=128,
    conv_dropout=0.25, dense_dropout=0.4,
    learning_rate=1e-3, l2_reg=1e-4,
)


##############################################################################
# ────────────────────────  SETUP HELPERS  ─────────────────────────────────
##############################################################################

def setup_seeds():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def setup_output_dirs(output_dir):
    for d in [output_dir,
              os.path.join(output_dir, "npy", "binary"),
              os.path.join(output_dir, "his", "binary"),
              os.path.join(output_dir, "checkpoints")]:
        os.makedirs(d, exist_ok=True)


##############################################################################
# ─────────────────────────  BASE CNN BACKBONE  ───────────────────────────
##############################################################################

def build_base_cnn(input_shape, n_classes, task,
                   filters_1=64, filters_2=128, filters_3=256,
                   dense_units_1=256, dense_units_2=128,
                   conv_dropout=0.25, dense_dropout=0.4,
                   learning_rate=1e-3, l2_reg=1e-4, **kwargs):
    """Plain Base CNN. Binary output. **kwargs absorbs unused DEFAULT_HP keys."""
    reg    = regularizers.l2(l2_reg)
    H, W   = input_shape[0], input_shape[1]
    inputs = keras.Input(shape=input_shape, name="input")

    x = layers.Conv2D(filters_1, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool1")(x); H //= 2; W //= 2
    x = layers.SpatialDropout2D(conv_dropout, name="sdrop1")(x)

    x = layers.Conv2D(filters_2, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    if H > 2 and W > 2:
        x = layers.MaxPooling2D((2, 2), name="pool2")(x); H //= 2; W //= 2
    x = layers.SpatialDropout2D(conv_dropout, name="sdrop2")(x)

    x = layers.Conv2D(filters_3, (3, 3), padding="same",
                      kernel_regularizer=reg, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dense(dense_units_1, kernel_regularizer=reg, name="dense1")(x)
    x = layers.BatchNormalization(name="bn_d1")(x)
    x = layers.ReLU(name="relu_d1")(x)
    x = layers.Dropout(dense_dropout, name="drop_d1")(x)

    x = layers.Dense(dense_units_2, kernel_regularizer=reg, name="dense2")(x)
    x = layers.BatchNormalization(name="bn_d2")(x)
    x = layers.ReLU(name="relu_d2")(x)
    x = layers.Dropout(dense_dropout, name="drop_d2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)
    model   = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc"),
                           keras.metrics.Precision(name="precision"),
                           keras.metrics.Recall(name="recall")])
    return model


##############################################################################
# ─────────────────────────  HP MECHANICS  ─────────────────────────────────
##############################################################################

def decode_position(position):
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


def full_cfg(tuned):
    c = dict(DEFAULT_HP)
    c.update(tuned)
    return c


def hp_objective(cfg, fold_data):
    """Single HP trial. Trains on the fold's training partition for up to
    20 epochs with early stopping, returns validation F1.
    """
    tf.keras.backend.clear_session()
    feature_shape = fold_data["feature_shape"]
    model = build_base_cnn(feature_shape, 2, "binary", **full_cfg(cfg))
    cw    = ip.get_class_weights(fold_data["y_int_train"], 2)
    model.fit(
        fold_data["X_train"], fold_data["y_train"],
        validation_data=(fold_data["X_val"], fold_data["y_val"]),
        epochs=20, batch_size=BATCH_SIZE, class_weight=cw,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True)],
        verbose=0)
    y_raw     = model.predict(fold_data["X_val"], verbose=0)
    y_pred, _ = ip.decode_predictions(y_raw, "binary")
    y_true    = fold_data["y_int_val"]
    return f1_score(y_true, y_pred, average="binary", zero_division=0)


##############################################################################
# ────────────────────  TUNER-SPECIFIC SAVE HELPERS  ──────────────────────
##############################################################################

def save_search_arrays(iter_scores, all_scores, pbest_val,
                       model_name, output_dir, ds_tag):
    npy_dir = os.path.join(output_dir, "npy", "binary")
    tag     = f"{model_name}_binary_{ds_tag}"
    np.save(os.path.join(npy_dir, f"{tag}_iter_scores.npy"), np.array(iter_scores))
    np.save(os.path.join(npy_dir, f"{tag}_all_scores.npy"),  np.array(all_scores))
    np.save(os.path.join(npy_dir, f"{tag}_pbest.npy"),       np.array(pbest_val))
    print(f"  Search arrays saved → npy/binary/{tag}_iter/all/pbest.npy")


def save_best_hps(best_cfg, model_name, output_dir, ds_tag):
    tag  = f"{model_name}_binary_{ds_tag}"
    path = os.path.join(output_dir, "npy", "binary", f"{tag}_best_hps.json")
    with open(path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"  Best HPs saved → npy/binary/{tag}_best_hps.json")


##############################################################################
# ──────────────────  EXPERIMENT + MAIN LOOP  ─────────────────────────────
##############################################################################

def run_experiment(dataset_name, model_name, output_dir, search_fn,
                   shared_search=None, source_dataset_name=None):
    """Run one binary experiment. If shared_search is None, invokes
    search_fn on fold 1 of the dataset. Otherwise reuses the transferred
    (best_cfg, ...) tuple.

    search_fn contract:
        search_fn(fold_data)
            → (best_cfg, iter_scores, all_scores, pbest_val)
    """
    ds_tag = ip.DATASET_TAG[dataset_name]
    print(f"\n{'='*65}\n  {dataset_name} | binary\n{'='*65}")

    # ── Search on fold 1, or transfer ────────────────────────────────────
    if shared_search is None:
        search_fold = ip.load_fold(dataset_name, "binary", 1)
        print(f"\n{'─'*50}\n  {model_name} HP Search (fold 1)\n{'─'*50}")
        t0 = time.time()
        best_cfg, iter_scores, all_scores, pbest_val = search_fn(search_fold)
        print(f"  Search done in {time.time()-t0:.1f}s")
    else:
        best_cfg, iter_scores, all_scores, pbest_val = shared_search
        print(f"\n{'─'*50}")
        print(f"  [Config transferred from {source_dataset_name} binary "
              f"{model_name} search — no new search]")
        print(f"  Using config: {best_cfg}")
        print(f"{'─'*50}")

    # ── CV + final via ablation_base ─────────────────────────────────────
    prefix = f"{model_name}_binary_{ds_tag}"
    t0     = time.time()
    (fold_m, test_m, final_h, cm,
     model, y_pred, y_prob, final) = ab.run_cv_and_final(
        build_base_cnn, full_cfg(best_cfg),
        dataset_name, "binary", output_dir, prefix)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Persist outputs ──────────────────────────────────────────────────
    ab.save_arrays(final["y_int_test"], y_pred, y_prob,
                   final["class_names"], model_name, "binary", ds_tag, output_dir)
    ab.save_history(final_h, model_name, "binary", ds_tag, output_dir)
    ab.save_cv_metrics(fold_m, model_name, "binary", ds_tag, output_dir)
    ab.save_metrics_table(test_m, model_name, "binary", ds_tag, output_dir)

    if shared_search is None:
        save_search_arrays(iter_scores, all_scores, pbest_val,
                           model_name, output_dir, ds_tag)
    save_best_hps(best_cfg, model_name, output_dir, ds_tag)

    summary = {
        "Dataset": dataset_name, "Task": "binary",
        f"{model_name}_Acc": test_m["Accuracy"],
        f"{model_name}_F1":  test_m["F1"],
        f"{model_name}_MCC": test_m["MCC"],
        f"{model_name}_AUC": test_m["AUC_ROC"],
        **{f"HP_{k}": v for k, v in best_cfg.items()},
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)
    print(f"  Summary saved. DONE: {dataset_name}")
    tf.keras.backend.clear_session()
    return summary, (best_cfg, iter_scores, all_scores, pbest_val)


def run_all_experiments(model_name, output_dir, search_fn, script_number):
    all_summaries  = []
    shared_search  = None
    source_dataset = None
    for ds in EXPERIMENT_PLAN:
        summary, search_results = run_experiment(
            ds, model_name, output_dir, search_fn,
            shared_search=shared_search,
            source_dataset_name=source_dataset)
        all_summaries.append(summary)
        if shared_search is None:
            shared_search  = search_results
            source_dataset = ds

    master = pd.DataFrame(all_summaries)
    master_path = os.path.join(
        output_dir, f"script{script_number}_master_summary.csv")
    master.to_csv(master_path, index=False)
    print(f"\nMaster summary saved.")
    print(master.to_string(index=False))
    print(f"\nSCRIPT {script_number} COMPLETE.")