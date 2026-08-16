##############################################################################
# ablation_base.py
#
# Shared training + save/plot infrastructure for the architectural ablation
# scripts (base CNN + scripts 6–11). All data loading goes through the
# precomputed folds/final from ids_pipeline — no scaling or resampling
# happens at run time.
#
# Each ablation script defines only:
#   - MODEL_NAME
#   - OUTPUT_DIR
#   - a build_model() callable
#   - DEFAULT_HP
# and calls ab.run_all_experiments(...).
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

from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import ids_pipeline as ip


##############################################################################
# ────────────────────────────  CONSTANTS  ─────────────────────────────────
##############################################################################

BATCH_SIZE   = 256
MAX_EPOCHS   = 25
RANDOM_STATE = 42

EXPERIMENT_PLAN = [
    #("CICIDS2017", "binary"),
    #("CICIDS2017", "multiclass"),
    ("UNSW_NB15",  "binary"),
    ("UNSW_NB15",  "multiclass"),
]


##############################################################################
# ─────────────────────────────  UTILITIES  ────────────────────────────────
##############################################################################

def setup_seeds():
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)


def setup_output_dirs(output_dir):
    for d in [output_dir,
              os.path.join(output_dir, "npy", "binary"),
              os.path.join(output_dir, "npy", "multiclass"),
              os.path.join(output_dir, "his", "binary"),
              os.path.join(output_dir, "his", "multiclass"),
              os.path.join(output_dir, "checkpoints")]:
        os.makedirs(d, exist_ok=True)


def get_callbacks(ckpt_path, patience_stop=10, patience_lr=5):
    return [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss",
            save_best_only=True, save_weights_only=True, verbose=0),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience_stop,
            restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=patience_lr, min_lr=1e-6, verbose=0),
    ]


##############################################################################
# ────────────────  CV + FINAL TRAINING (uses precomputed data)  ──────────
##############################################################################

def run_cv_and_final(model_builder, build_kwargs,
                     dataset_name, task, output_dir, prefix):
    """5-fold CV over precomputed folds, then final training on the
    precomputed 80% train partition. Test set is scaler-transformed but
    otherwise untouched.
    """
    final         = ip.load_final(dataset_name, task)
    feature_shape = final["feature_shape"]
    n_classes     = final["n_classes"]
    ckpt_base     = os.path.join(output_dir, "checkpoints")
    fold_metrics  = []

    # ── 5-Fold CV over precomputed folds ─────────────────────────────────
    print(f"\n  [{prefix}] 5-fold CV (precomputed)...")
    for fold_idx, fold in ip.iter_folds(dataset_name, task):
        model = model_builder(feature_shape, n_classes, task, **build_kwargs)
        cw    = ip.get_class_weights(fold["y_int_train"], n_classes)
        ckpt  = os.path.join(ckpt_base, f"{prefix}_fold{fold_idx}.weights.h5")
        model.fit(
            fold["X_train"], fold["y_train"],
            validation_data=(fold["X_val"], fold["y_val"]),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
            class_weight=cw, callbacks=get_callbacks(ckpt), verbose=0)

        y_raw          = model.predict(fold["X_val"], verbose=0)
        y_pred, y_prob = ip.decode_predictions(y_raw, task)
        y_true         = fold["y_int_val"]
        m = ip.compute_metrics(y_true, y_pred, y_prob, task)
        m["Fold"] = fold_idx
        fold_metrics.append(m)
        print(f"    Fold {fold_idx}: Acc={m['Accuracy']:.4f}  "
              f"F1={m['F1']:.4f}  MCC={m['MCC']:.4f}  "
              f"AUC={m['AUC_ROC']:.4f}")
        tf.keras.backend.clear_session()

    # ── Final training on precomputed 80% train ──────────────────────────
    print(f"\n  [{prefix}] Final training...")
    cw    = ip.get_class_weights(final["y_int_train"], n_classes)
    model = model_builder(feature_shape, n_classes, task, **build_kwargs)
    ckpt  = os.path.join(ckpt_base, f"{prefix}_final.weights.h5")
    final_history = model.fit(
        final["X_train"], final["y_train"],
        validation_split=0.1,
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        class_weight=cw, callbacks=get_callbacks(ckpt), verbose=1)
    if os.path.exists(ckpt):
        model.load_weights(ckpt)

    y_raw          = model.predict(final["X_test"], verbose=0)
    y_pred, y_prob = ip.decode_predictions(y_raw, task)
    final_metrics  = ip.compute_metrics(
        final["y_int_test"], y_pred, y_prob, task)
    cm             = confusion_matrix(final["y_int_test"], y_pred)

    print(f"\n  Test → Acc={final_metrics['Accuracy']:.4f}  "
          f"F1={final_metrics['F1']:.4f}  "
          f"MCC={final_metrics['MCC']:.4f}  "
          f"AUC={final_metrics['AUC_ROC']:.4f}")

    return (fold_metrics, final_metrics, final_history,
            cm, model, y_pred, y_prob, final)


##############################################################################
# ─────────────────────────  SAVE HELPERS  ─────────────────────────────────
##############################################################################

def save_arrays(y_true, y_pred, y_prob, class_names,
                model_name, task, ds_tag, output_dir):
    npy_dir = os.path.join(output_dir, "npy", task)
    tag     = f"{model_name}_{task}_{ds_tag}"
    np.save(os.path.join(npy_dir, f"{tag}_y_true.npy"),      y_true)
    np.save(os.path.join(npy_dir, f"{tag}_y_pred.npy"),      y_pred)
    np.save(os.path.join(npy_dir, f"{tag}_y_prob.npy"),      y_prob)
    np.save(os.path.join(npy_dir, f"{tag}_class_names.npy"), np.array(class_names))
    print(f"  Arrays saved → npy/{task}/{tag}_*.npy")


def save_history(history, model_name, task, ds_tag, output_dir):
    his_dir = os.path.join(output_dir, "his", task)
    tag     = f"{model_name}_{task}_{ds_tag}"
    h = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(his_dir, f"{tag}_history.json"), "w") as f:
        json.dump(h, f, indent=2)
    print(f"  History saved → his/{task}/{tag}_history.json")


def save_cv_metrics(fold_metrics, model_name, task, ds_tag, output_dir):
    tag  = f"{model_name}_{task}_{ds_tag}"
    path = os.path.join(output_dir, f"{tag}_cv_metrics.csv")
    pd.DataFrame(fold_metrics).to_csv(path, index=False)
    print(f"  CV metrics saved → {tag}_cv_metrics.csv")


def save_confusion_matrix(cm, class_names, model_name, task, ds_tag, output_dir):
    n   = len(class_names)
    raw = cm.astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.nan_to_num(raw / raw.sum(axis=1, keepdims=True))
    cell = max(0.7, min(1.4, 8 / n)); fsz = max(5, min(9, 80 // n))
    fig, ax = plt.subplots(figsize=(max(5, n * cell + 1), max(5, n * cell + 1)))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=fsz)
    ax.set_yticklabels(class_names, fontsize=fsz)
    thresh = norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{norm[i,j]:.2f}\n({raw[i,j]})",
                    ha="center", va="center", fontsize=max(4, fsz - 1),
                    color="white" if norm[i, j] > thresh else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    fname = f"{model_name}_{task}_{ds_tag}_CM.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.close()
    print(f"  Saved: {fname}")


def save_roc_curves(y_true, y_prob, class_names, model_name, task, ds_tag, output_dir):
    n_classes = len(class_names)
    fig, ax   = plt.subplots(figsize=(8, 6))
    y_bin = label_binarize(y_true, classes=np.arange(n_classes))
    cmap  = matplotlib.colormaps.get_cmap("tab20").resampled(n_classes)
    fprs, tprs = [], []
    for i, name in enumerate(class_names):
        if y_bin[:, i].sum() == 0:
            continue
        fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fprs.append(fpr_i); tprs.append(tpr_i)
        ax.plot(fpr_i, tpr_i, color=cmap(i), lw=1.2, alpha=0.75,
                label=f"{name} ({auc(fpr_i, tpr_i):.3f})")
    all_fpr  = np.unique(np.concatenate(fprs))
    mean_tpr = np.mean(
        [np.interp(all_fpr, fp, tp) for fp, tp in zip(fprs, tprs)], axis=0)
    ax.plot(all_fpr, mean_tpr, "k--", lw=2.5,
            label=f"Macro-avg ({auc(all_fpr, mean_tpr):.4f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", alpha=0.6)
    ax.set_xlim([-0.01, 1.01]); ax.set_ylim([-0.01, 1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{model_name}_{task}_{ds_tag}_ROC.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.close()
    print(f"  Saved: {fname}")


def save_metrics_table(metrics, model_name, task, ds_tag, output_dir):
    rows = [[k, f"{v:.4f}"] for k, v in metrics.items() if k != "Fold"]
    fig, ax = plt.subplots(figsize=(5, max(2, len(rows) * 0.45 + 0.8)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.3, 1.4)
    for j in range(2):
        tbl[0, j].set_facecolor("#2E86AB")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    plt.tight_layout()
    fname = f"{model_name}_{task}_{ds_tag}_MT.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=800)
    plt.close()
    print(f"  Saved: {fname}")


##############################################################################
# ────────────────────────  EXPERIMENT RUNNER  ────────────────────────────
##############################################################################

def run_experiment(model_builder, build_kwargs, model_name,
                   dataset_name, task, output_dir):
    """Run one (dataset, task) experiment end-to-end. Returns the summary
    dict that gets aggregated by run_all_experiments.
    """
    ds_tag = ip.DATASET_TAG[dataset_name]
    print(f"\n{'='*65}")
    print(f"  {dataset_name} | {task}")
    print(f"{'='*65}")

    prefix = f"{model_name}_{task}_{ds_tag}"
    t0     = time.time()

    (fold_m, test_m, final_h, cm,
     model, y_pred, y_prob, final) = run_cv_and_final(
        model_builder, build_kwargs, dataset_name, task, output_dir, prefix)

    print(f"  Done in {time.time()-t0:.1f}s")

    save_arrays(final["y_int_test"], y_pred, y_prob,
                final["class_names"], model_name, task, ds_tag, output_dir)
    save_history(final_h, model_name, task, ds_tag, output_dir)
    save_cv_metrics(fold_m, model_name, task, ds_tag, output_dir)

    if task == "multiclass":
        save_confusion_matrix(cm, final["class_names"],
                              model_name, task, ds_tag, output_dir)
        save_roc_curves(final["y_int_test"], y_prob, final["class_names"],
                        model_name, task, ds_tag, output_dir)

    save_metrics_table(test_m, model_name, task, ds_tag, output_dir)

    summary = {
        "Dataset": dataset_name, "Task": task,
        f"{model_name}_Acc": test_m["Accuracy"],
        f"{model_name}_F1":  test_m["F1"],
        f"{model_name}_MCC": test_m["MCC"],
        f"{model_name}_AUC": test_m["AUC_ROC"],
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)
    print(f"  Summary saved. DONE: {dataset_name} | {task}")
    tf.keras.backend.clear_session()
    return summary


def run_all_experiments(model_builder, build_kwargs, model_name,
                        output_dir, master_filename):
    """Iterate the 4-variant EXPERIMENT_PLAN, aggregate summaries, save
    master CSV.
    """
    all_summaries = []
    for ds, task in EXPERIMENT_PLAN:
        all_summaries.append(
            run_experiment(model_builder, build_kwargs, model_name,
                           ds, task, output_dir))
    master = pd.DataFrame(all_summaries)
    master.to_csv(os.path.join(output_dir, master_filename), index=False)
    print(f"\nMaster summary saved: {master_filename}")
    print(master.to_string(index=False))