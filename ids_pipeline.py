##############################################################################
# ids_pipeline.py — Two-phase data pipeline
#
# PHASE 1 (one-time): python ids_pipeline.py
#   For each (dataset, task) variant:
#     - 80/20 stratified split on cleaned data (no scaling, no resampling)
#     - For each of 5 CV folds:
#         fold-train:  SMOTE-Tomek → MinMax scale, saved
#         fold-val:    MinMax scale only (real data), saved
#     - Final: 80% train SMOTE-Tomek → MinMax scale, saved
#              20% test scaler-transformed (untouched), saved
#     - Pre/post-resample counts JSON written for prep.py
#
# PHASE 2 (training scripts):
#   from ids_pipeline import iter_folds, load_final
#   for fold_idx, fold in iter_folds(ds, task):
#       ...   # fold has X_train, y_train, y_int_train, X_val, y_val, y_int_val
#   final = load_final(ds, task)
#   ...       # final has X_train, y_train, y_int_train, X_test, y_test, ...

"""
usage:
    python ids_pipeline.py
    # Prepares all 4 (dataset, task) variants and saves to disk.

"""
##############################################################################

import os
import json
import math
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek


##############################################################################
# ────────────────────────────  CONFIGURATION  ─────────────────────────────
##############################################################################

BASE = "/mnt/c/Users/Battosai Himura/Desktop/Projects/tf-gpu-env/ids"

CLEAN_PATHS = {
    "CICIDS2017": f"{BASE}/data/preprocessed/cicids_preprocessed.csv",
    "UNSW_NB15":  f"{BASE}/data/preprocessed/unsw-nb15_preprocessed.csv",
}
LABEL_COLS = {
    "CICIDS2017": {"binary": "b_label", "multiclass": " Label"},
    "UNSW_NB15":  {"binary": "label",   "multiclass": "attack_cat"},
}
ALL_LABEL_COLS = {"b_label", " Label", "label", "attack_cat"}
DATASET_TAG    = {"CICIDS2017": "cicids", "UNSW_NB15": "unsw"}

PROCESSED_DIR    = f"{BASE}/data/processed"
RESAMPLE_LOG_DIR = f"{BASE}/data/resample_logs"
CACHE_DIR        = f"{BASE}/data/pipeline_cache"

CICIDS_CLASS_CAP        = 310_000
UNSW_MC_MAJORITY_CAP    = 200_000
UNSW_MC_MINORITY_RATIO  = 0.5
CICIDS_MC_BALANCE_CAP   = None

N_SPLITS = 5
SEED     = 42

for d in [PROCESSED_DIR, RESAMPLE_LOG_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


##############################################################################
# ───────────────────  DATA LOADING + CICIDS SHRINKAGE  ────────────────────
##############################################################################

def _shrink_by_random_cap(df, label_col, class_cap, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for cls, group in df.groupby(label_col, sort=False):
        if len(group) > class_cap:
            picks = rng.choice(group.index.values, size=class_cap, replace=False)
            keep.extend(picks)
        else:
            keep.extend(group.index.values)
    return df.loc[np.sort(keep)].reset_index(drop=True)


def _cicids_cache_path(class_cap, seed):
    return os.path.join(CACHE_DIR,
                        f"cicids_clean_cap{class_cap}_seed{seed}.parquet")


def load_cleaned_source(dataset_name, use_cache=True):
    path = CLEAN_PATHS[dataset_name]
    if dataset_name == "CICIDS2017":
        cache_path = _cicids_cache_path(CICIDS_CLASS_CAP, SEED)
        if use_cache and os.path.exists(cache_path):
            print(f"[pipeline] Loading cached shrunk CICIDS: {cache_path}")
            return pd.read_parquet(cache_path)
        print(f"[pipeline] Loading clean CICIDS: {path}")
        df = pd.read_csv(path, low_memory=False)
        df[" Label"] = df[" Label"].astype(str).str.strip()
        df = _shrink_by_random_cap(df, " Label", CICIDS_CLASS_CAP, SEED)
        print(f"[pipeline]   shrunk shape: {df.shape}")
        if use_cache:
            df.to_parquet(cache_path, index=False)
        return df
    print(f"[pipeline] Loading clean UNSW: {path}")
    df = pd.read_csv(path, low_memory=False)
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    return df


##############################################################################
# ─────────────────────  FEATURE / LABEL EXTRACTION  ──────────────────────
##############################################################################

def _compute_2d_shape(n_features):
    k = math.ceil(math.sqrt(n_features))
    return k, k


def prepare_features_labels(df, dataset_name, task):
    label_col = LABEL_COLS[dataset_name][task]
    feature_cols = [c for c in df.columns if c not in ALL_LABEL_COLS]
    X = df[feature_cols].values.astype(np.float32)
    if task == "binary":
        y_int = df[label_col].values.astype(np.int32)
        y_encoded = y_int.reshape(-1, 1).astype(np.float32)
        class_names = ["BENIGN", "ATTACK"]
    else:
        le = LabelEncoder()
        y_int = le.fit_transform(df[label_col].values.astype(str)).astype(np.int32)
        ohe = OneHotEncoder(sparse_output=False)
        y_encoded = ohe.fit_transform(y_int.reshape(-1, 1)).astype(np.float32)
        class_names = list(le.classes_)
    n_features = X.shape[1]
    H, W = _compute_2d_shape(n_features)
    pad = H * W - n_features
    if pad > 0:
        X = np.concatenate([X, np.zeros((len(X), pad), dtype=np.float32)], axis=1)
    return X, y_int, y_encoded, class_names, (H, W, 1)


##############################################################################
# ───────────────────────  RESAMPLER CONSTRUCTION  ─────────────────────────
##############################################################################

def _sampling_strategy_for(dataset_name, task, y_int_train):
    counts = pd.Series(y_int_train).value_counts().to_dict()
    if task == "binary":
        target = max(counts.values())
        return ({c: min(counts[c], target) for c in counts},
                {c: target for c in counts})
    if dataset_name == "CICIDS2017":
        target = CICIDS_MC_BALANCE_CAP or max(counts.values())
        return ({c: min(counts[c], target) for c in counts},
                {c: target for c in counts})
    # UNSW multiclass — partial balance
    maj_cls = max(counts, key=counts.get)
    minority_target = int(UNSW_MC_MINORITY_RATIO * UNSW_MC_MAJORITY_CAP)
    under, smote = {}, {}
    for c in counts:
        if c == maj_cls:
            under[c] = min(counts[c], UNSW_MC_MAJORITY_CAP)
            smote[c] = UNSW_MC_MAJORITY_CAP
        else:
            under[c] = min(counts[c], minority_target)
            smote[c] = minority_target
    return under, smote


def _make_resampler(dataset_name, task, y_int_train, seed=SEED):
    under_dict, smote_dict = _sampling_strategy_for(dataset_name, task, y_int_train)
    orig_counts = pd.Series(y_int_train).value_counts().to_dict()
    counts_after_under = {c: min(orig_counts[c], under_dict[c]) for c in under_dict}
    min_class_size = min(counts_after_under.values())
    k_neighbors = max(1, min(5, min_class_size - 1))
    nn_k = NearestNeighbors(n_neighbors=k_neighbors + 1, n_jobs=-1)
    smote = SMOTE(sampling_strategy=smote_dict, k_neighbors=nn_k, random_state=seed)
    smote_tomek = SMOTETomek(sampling_strategy=smote_dict, smote=smote, random_state=seed)
    return ImbPipeline([
        ("under", RandomUnderSampler(sampling_strategy=under_dict, random_state=seed)),
        ("smt",   smote_tomek),
    ])


##############################################################################
# ──────────────────────  PATH HELPERS FOR PROCESSED  ─────────────────────
##############################################################################

def _variant_dir(dataset_name, task):
    return os.path.join(PROCESSED_DIR, f"{DATASET_TAG[dataset_name]}_{task}")

def _final_dir(dataset_name, task):
    return os.path.join(_variant_dir(dataset_name, task), "final")

def _fold_dir(dataset_name, task, fold_idx):
    return os.path.join(_variant_dir(dataset_name, task), "folds", f"fold_{fold_idx}")


##############################################################################
# ────────────────────────  RESAMPLE COUNT LOGGING  ────────────────────────
##############################################################################

def _log_resample_counts(dataset_name, task, y_int_before, y_int_after, class_names):
    tag = DATASET_TAG[dataset_name]
    before = pd.Series(y_int_before).value_counts().to_dict()
    after  = pd.Series(y_int_after).value_counts().to_dict()
    def to_named(d):
        return {class_names[i]: int(d.get(i, 0)) for i in range(len(class_names))}
    payload = dict(
        dataset=dataset_name, task=task,
        source="final_80pct_train",
        before=to_named(before), after=to_named(after),
    )
    out = os.path.join(RESAMPLE_LOG_DIR, f"{tag}_{task}_counts.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)


##############################################################################
# ───────────────────  PHASE 1: ONE-TIME PREPARATION  ─────────────────────
##############################################################################

def _encode(y_int, n_classes, task):
    if task == "binary":
        return y_int.reshape(-1, 1).astype(np.float32)
    return np.eye(n_classes, dtype=np.float32)[y_int]


def _resample_scale(X_train_flat, y_int_train, dataset_name, task):
    """SMOTE-Tomek then MinMaxScaler, both fit on training data only."""
    resampler = _make_resampler(dataset_name, task, y_int_train)
    X_bal_flat, y_int_bal = resampler.fit_resample(X_train_flat, y_int_train)
    scaler = MinMaxScaler()
    X_bal_scaled_flat = scaler.fit_transform(X_bal_flat)
    return X_bal_scaled_flat, y_int_bal, scaler


def _reshape(X_flat, feature_shape):
    return X_flat.reshape((-1,) + feature_shape)


def prepare_variant(dataset_name, task):
    print(f"\n{'='*70}\n[prepare] {dataset_name} | {task}\n{'='*70}")

    df = load_cleaned_source(dataset_name)
    X_flat, y_int, y_enc, class_names, feature_shape = \
        prepare_features_labels(df, dataset_name, task)
    n_classes = len(class_names)
    print(f"[prepare]   samples: {len(X_flat)}, features → {feature_shape}")

    # Write shared meta
    variant_dir = _variant_dir(dataset_name, task)
    os.makedirs(variant_dir, exist_ok=True)
    np.save(os.path.join(variant_dir, "class_names.npy"), np.array(class_names))
    with open(os.path.join(variant_dir, "feature_shape.json"), "w") as f:
        json.dump(list(feature_shape), f)

    # 80/20 stratified split
    idx = np.arange(len(X_flat))
    tr_idx, te_idx = train_test_split(
        idx, test_size=0.2, stratify=y_int, random_state=SEED)
    X_tr_flat, X_te_flat = X_flat[tr_idx], X_flat[te_idx]
    y_int_tr, y_int_te   = y_int[tr_idx], y_int[te_idx]
    print(f"[prepare]   80/20 split → train {len(X_tr_flat)}, test {len(X_te_flat)}")

    # FINAL
    print(f"[prepare]   [final]  SMOTE-Tomek + MinMax on 80% train...")
    X_tr_s_flat, y_int_tr_bal, scaler = _resample_scale(
        X_tr_flat, y_int_tr, dataset_name, task)
    X_te_s_flat = scaler.transform(X_te_flat)
    print(f"[prepare]   [final]  train after resample: {len(X_tr_s_flat)}")

    final_dir = _final_dir(dataset_name, task)
    os.makedirs(final_dir, exist_ok=True)
    np.save(os.path.join(final_dir, "X_train.npy"),
            _reshape(X_tr_s_flat, feature_shape))
    np.save(os.path.join(final_dir, "y_train.npy"),
            _encode(y_int_tr_bal, n_classes, task))
    np.save(os.path.join(final_dir, "y_train_int.npy"), y_int_tr_bal)
    np.save(os.path.join(final_dir, "X_test.npy"),
            _reshape(X_te_s_flat, feature_shape))
    np.save(os.path.join(final_dir, "y_test.npy"),
            _encode(y_int_te, n_classes, task))
    np.save(os.path.join(final_dir, "y_test_int.npy"), y_int_te)
    with open(os.path.join(final_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"[prepare]   [final]  saved → {final_dir}/")

    _log_resample_counts(dataset_name, task, y_int_tr, y_int_tr_bal, class_names)

    # FOLDS
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold_idx, (fold_tr, fold_val) in enumerate(
            skf.split(X_tr_flat, y_int_tr), start=1):
        print(f"[prepare]   [fold {fold_idx}/{N_SPLITS}]  "
              f"train {len(fold_tr)}, val {len(fold_val)}")
        Xf_tr_flat  = X_tr_flat[fold_tr]
        yf_int_tr   = y_int_tr[fold_tr]
        Xf_val_flat = X_tr_flat[fold_val]
        yf_int_val  = y_int_tr[fold_val]

        Xf_tr_s_flat, yf_int_tr_bal, fold_scaler = _resample_scale(
            Xf_tr_flat, yf_int_tr, dataset_name, task)
        Xf_val_s_flat = fold_scaler.transform(Xf_val_flat)
        print(f"[prepare]   [fold {fold_idx}/{N_SPLITS}]  "
              f"fold-train after resample: {len(Xf_tr_s_flat)}")

        fdir = _fold_dir(dataset_name, task, fold_idx)
        os.makedirs(fdir, exist_ok=True)
        np.save(os.path.join(fdir, "X_train.npy"),
                _reshape(Xf_tr_s_flat, feature_shape))
        np.save(os.path.join(fdir, "y_train.npy"),
                _encode(yf_int_tr_bal, n_classes, task))
        np.save(os.path.join(fdir, "y_train_int.npy"), yf_int_tr_bal)
        np.save(os.path.join(fdir, "X_val.npy"),
                _reshape(Xf_val_s_flat, feature_shape))
        np.save(os.path.join(fdir, "y_val.npy"),
                _encode(yf_int_val, n_classes, task))
        np.save(os.path.join(fdir, "y_val_int.npy"), yf_int_val)
        with open(os.path.join(fdir, "scaler.pkl"), "wb") as f:
            pickle.dump(fold_scaler, f)

    print(f"[prepare]   done: {variant_dir}/")


##############################################################################
# ────────────────────  PHASE 2: LOADERS FOR SCRIPTS  ─────────────────────
##############################################################################

def _load_variant_meta(dataset_name, task):
    variant_dir = _variant_dir(dataset_name, task)
    if not os.path.exists(variant_dir):
        raise FileNotFoundError(
            f"Processed data not found at {variant_dir}. "
            f"Run 'python ids_pipeline.py' first.")
    with open(os.path.join(variant_dir, "feature_shape.json")) as f:
        feature_shape = tuple(json.load(f))
    class_names = np.load(
        os.path.join(variant_dir, "class_names.npy"),
        allow_pickle=True).tolist()
    return class_names, feature_shape


def load_final(dataset_name, task):
    class_names, feature_shape = _load_variant_meta(dataset_name, task)
    d = _final_dir(dataset_name, task)
    with open(os.path.join(d, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return dict(
        dataset_name=dataset_name, task=task,
        class_names=class_names, feature_shape=feature_shape,
        n_classes=len(class_names), scaler=scaler,
        X_train    =np.load(os.path.join(d, "X_train.npy")),
        y_train    =np.load(os.path.join(d, "y_train.npy")),
        y_int_train=np.load(os.path.join(d, "y_train_int.npy")),
        X_test     =np.load(os.path.join(d, "X_test.npy")),
        y_test     =np.load(os.path.join(d, "y_test.npy")),
        y_int_test =np.load(os.path.join(d, "y_test_int.npy")),
    )


def load_fold(dataset_name, task, fold_idx):
    class_names, feature_shape = _load_variant_meta(dataset_name, task)
    d = _fold_dir(dataset_name, task, fold_idx)
    if not os.path.exists(d):
        raise FileNotFoundError(f"Fold {fold_idx} not found at {d}.")
    with open(os.path.join(d, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    return dict(
        dataset_name=dataset_name, task=task, fold_idx=fold_idx,
        class_names=class_names, feature_shape=feature_shape,
        n_classes=len(class_names), scaler=scaler,
        X_train    =np.load(os.path.join(d, "X_train.npy")),
        y_train    =np.load(os.path.join(d, "y_train.npy")),
        y_int_train=np.load(os.path.join(d, "y_train_int.npy")),
        X_val      =np.load(os.path.join(d, "X_val.npy")),
        y_val      =np.load(os.path.join(d, "y_val.npy")),
        y_int_val  =np.load(os.path.join(d, "y_val_int.npy")),
    )


def iter_folds(dataset_name, task, n_splits=N_SPLITS):
    for fold_idx in range(1, n_splits + 1):
        yield fold_idx, load_fold(dataset_name, task, fold_idx)


##############################################################################
# ─────────────────────  SHARED HELPERS FOR TRAINING  ─────────────────────
##############################################################################

def get_class_weights(y_int, n_classes):
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_int)
    return {i: float(v) for i, v in enumerate(w)}


def decode_predictions(y_raw, task):
    if task == "binary":
        prob = y_raw.ravel()
        return (prob >= 0.5).astype(int), prob
    return np.argmax(y_raw, axis=1), y_raw


def compute_metrics(y_true, y_pred, y_prob, task):
    avg = "binary" if task == "binary" else "macro"
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_true, y_pred, average=avg, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)
    try:
        auc_roc = (roc_auc_score(y_true, y_prob) if task == "binary" else
                   roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc_roc = float("nan")
    return dict(Accuracy=acc, Precision=prec, Recall=rec,
                F1=f1, MCC=mcc, AUC_ROC=auc_roc)


##############################################################################
# ────────────  CLI: `python ids_pipeline.py` prepares all 4  ─────────────
##############################################################################

def prepare_all():
    for ds, task in [
        ("CICIDS2017", "binary"),
        ("CICIDS2017", "multiclass"),
        ("UNSW_NB15",  "binary"),
        ("UNSW_NB15",  "multiclass"),
    ]:
        prepare_variant(ds, task)
    print("\n[pipeline] Done. All 4 variants processed and saved.")


if __name__ == "__main__":
    prepare_all()