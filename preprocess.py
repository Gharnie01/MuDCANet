#!/usr/bin/env python3
##############################################################################
# preprocess.py — CLI cleaning script for UNSW-NB15 and CIC-IDS2017.
#
# Produces the cleaned CSVs consumed by ids_pipeline / MuDCANet.py training.
# Does NOT scale or resample. Scaling (MinMax) and resampling (SMOTETomek)
# are handled per fold inside ids_pipeline to keep test-set information out
# of training.
#
# Steps performed per dataset:
#
#   UNSW-NB15:
#     load → drop nulls → drop duplicates → strip whitespace on attack_cat
#     → one-hot encode proto/service/state → remove 'Worms' category
#     → save cleaned CSV (both `label` and `attack_cat` columns retained)
#
#   CIC-IDS2017:
#     combine input CSVs → add `b_label` (0/1) → drop nulls
#     → replace inf with NaN and drop → drop duplicates
#     → normalize label column name to ` Label` (with leading space)
#     → strip whitespace on ` Label` → consolidate attack classes
#     → save cleaned CSV (both `b_label` and ` Label` columns retained)
#
# Optional EDA plots show class distributions before and after cleaning
# (stage 1 and stage 2). The stage-4 SMOTEENN plot from the old script is
# gone — that role now belongs to prep.py, which reads from the training
# pipeline's per-run JSON logs.
#
# Usage:
#     python preprocess.py --dataset both
#     python preprocess.py --dataset unsw --no-visuals
#     python preprocess.py --dataset cicids --force
##############################################################################

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##############################################################################
# ─────────────────────────────  DEFAULT PATHS  ────────────────────────────
##############################################################################
# All defaults reflect the layout used elsewhere in the project. Override
# any of them via CLI flags.

DEFAULT_UNSW_INPUT   = (
    "/mnt/c/Users/Battosai Himura/Desktop/Projects/tf-gpu-env/ids/data/raw/raw_unsw-nb15/UNSW-NB15-full.csv"
)

DEFAULT_CICIDS_INPUT = (
    "/mnt/c/Users/Battosai Himura/Desktop/Projects/tf-gpu-env/ids/data/raw/raw_cicids2017"
)

DEFAULT_OUTPUT_BASE = (
    "/mnt/c/Users/Battosai Himura/Desktop/Projects/tf-gpu-env/ids/data"
)

UNSW_OUT_SUBDIR   = "preprocessed"
CICIDS_OUT_SUBDIR = "preprocessed"

UNSW_VIS_SUBDIR   = "dataset_visuals/unsw/"
CICIDS_VIS_SUBDIR = "dataset_visuals/cicids/"

UNSW_OUT_FILENAME   = "unsw-nb15_preprocessed.csv"
CICIDS_OUT_FILENAME = "cicids_preprocessed.csv"


##############################################################################
# ────────────────────────────  CICIDS LABEL MAP  ──────────────────────────
##############################################################################
# Attack-class consolidation for CIC-IDS2017. Preserved verbatim from the
# original preprocess.py.

CICIDS_LABEL_MAP = {
    'FTP-Patator':               'Brute Force',
    'SSH-Patator':               'Brute Force',
    'DDoS':                      'DoS/DDoS',
    'DoS GoldenEye':             'DoS/DDoS',
    'DoS Hulk':                  'DoS/DDoS',
    'DoS Slowhttptest':          'DoS/DDoS',
    'DoS slowloris':             'DoS/DDoS',
    'Heartbleed':                'DoS/DDoS',
    'Web Attack � Brute Force':  'Web Attack',
    'Web Attack � Sql Injection':'Web Attack',
    'Web Attack � XSS':          'Web Attack',
}


##############################################################################
# ───────────────────────────  EDA PLOT HELPER  ────────────────────────────
##############################################################################

def plot_distribution(counts, xlabel, ylabel, filename, color=None):
    """Bar chart with counts + percentages annotated above each bar.
    Preserved from the original preprocess.py."""
    default_colors = plt.cm.tab10.colors
    bar_colors = (color if color else
                  [default_colors[i % len(default_colors)]
                   for i in range(len(counts))])

    fig, ax = plt.subplots(figsize=(max(8, len(counts) * 1.2), 6))
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=bar_colors, edgecolor='black', linewidth=0.5)

    total = counts.sum()
    for bar, value in zip(bars, counts.values):
        pct = f"{value / total * 100:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:,}\n({pct})",
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_ylim(0, counts.max() * 1.12)

    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=500)
    plt.close()
    print(f"    Saved: {filename}")


##############################################################################
# ─────────────────────────────  UNSW PIPELINE  ────────────────────────────
##############################################################################

def preprocess_unsw(input_path, output_dir, visuals_dir, make_visuals=True):
    """Clean UNSW-NB15 and save. Returns the cleaned DataFrame."""
    print(f"\n{'='*70}")
    print(f"  UNSW-NB15 preprocessing")
    print(f"{'='*70}")
    print(f"  Input : {input_path}")
    print(f"  Output: {os.path.join(output_dir, UNSW_OUT_FILENAME)}")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"UNSW input not found: {input_path}")

    print(f"\n  Reading...")
    df = pd.read_csv(input_path, low_memory=False)
    initial_shape = df.shape
    print(f"    Loaded: {df.shape}")

    # ─── Stage 1 EDA (before cleaning) ────────────────────────────────────
    if make_visuals:
        print(f"\n  Stage 1 EDA (before cleaning)...")
        mc_counts = df['attack_cat'].astype(str).str.strip().value_counts()
        plot_distribution(
            mc_counts,
            xlabel="Attack Category", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage1_multiclass.png"),
        )
        bin_counts = df['label'].value_counts().sort_index()
        plot_distribution(
            bin_counts,
            xlabel="Label (0 = Benign, 1 = Attack)", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage1_binary.png"),
            color=['skyblue', 'lightcoral'],
        )

    # ─── Cleaning ──────────────────────────────────────────────────────────
    print(f"\n  Cleaning...")

    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"    Nulls in {(null_counts > 0).sum()} column(s); dropping rows.")
    df.dropna(inplace=True)
    print(f"    Shape after nulls: {df.shape}")

    dupes = df.duplicated().sum()
    print(f"    Duplicate rows: {dupes}")
    df.drop_duplicates(inplace=True)
    print(f"    Shape after dupes: {df.shape}")

    df['attack_cat'] = df['attack_cat'].astype(str).str.strip()

    cat_cols = ['proto', 'service', 'state']
    print(f"\n    One-hot cardinality before encoding:")
    for col in cat_cols:
        print(f"      {col}: {df[col].nunique()} unique values")
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    print(f"    Shape after one-hot: {df.shape}")

    # ─── Rare-class removal (BEFORE save — differs from original script) ─
    # Under the new pipeline, ids_pipeline reads this CSV directly and does
    # not do any class filtering, so we drop the excluded classes here.
    # Worms, DoS, and Shellcode are removed for the same reason: too few
    # samples to support faithful fold-wise resampling without generating
    # overwhelmingly synthetic training data for those classes.
    UNSW_DROP_CLASSES = ['Worms', 'DoS', 'Shellcode']
    before_drop = len(df)
    df = df[~df['attack_cat'].isin(UNSW_DROP_CLASSES)].reset_index(drop=True)
    print(f"    Dropped classes {UNSW_DROP_CLASSES}: "
          f"{before_drop - len(df)} rows removed")
    print(f"    Final shape: {df.shape}")
    print(f"    Total rows removed overall: {initial_shape[0] - df.shape[0]}")

    # ─── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, UNSW_OUT_FILENAME)
    df.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")

    # ─── Stage 2 EDA (after cleaning) ──────────────────────────────────────
    if make_visuals:
        print(f"\n  Stage 2 EDA (after cleaning)...")
        mc_counts_clean = df['attack_cat'].value_counts()
        plot_distribution(
            mc_counts_clean,
            xlabel="Attack Category", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage2_multiclass.png"),
        )
        bin_counts_clean = df['label'].value_counts().sort_index()
        plot_distribution(
            bin_counts_clean,
            xlabel="Label (0 = Benign, 1 = Attack)", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage2_binary.png"),
            color=['skyblue', 'lightcoral'],
        )

    return df


##############################################################################
# ───────────────────────────  CIC-IDS PIPELINE  ───────────────────────────
##############################################################################

def combine_cicids_csvs(input_folder):
    """Concatenate all CSVs in the given folder into a single DataFrame.
    Exported for reuse from Jupyter/IPython.
    """
    pattern   = os.path.join(input_folder, "*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSVs found in: {input_folder}\n"
            f"Pattern searched: {pattern}")

    print(f"    Found {len(csv_files)} file(s):")
    dfs = []
    for f in csv_files:
        print(f"      {os.path.basename(f)}")
        df_temp = pd.read_csv(f, low_memory=False)
        print(f"        → {df_temp.shape[0]:,} rows × {df_temp.shape[1]} cols")
        dfs.append(df_temp)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"    Combined shape: {combined.shape}")
    return combined


def _normalize_cicids_label_column(df):
    """CIC-IDS files sometimes have `Label` (no leading space) and
    sometimes ` Label` (with leading space). ids_pipeline expects
    ` Label`, so rename to that if needed. Returns the resolved column
    name for use in this function only (always ` Label` afterwards).
    """
    if ' Label' in df.columns:
        return ' Label'
    if 'Label' in df.columns:
        df.rename(columns={'Label': ' Label'}, inplace=True)
        print(f"    Renamed column 'Label' → ' Label' (leading space)")
        return ' Label'
    raise KeyError(
        "Neither 'Label' nor ' Label' found in CIC-IDS data. "
        f"Available columns: {list(df.columns)[:10]}...")


def preprocess_cicids(input_folder, output_dir, visuals_dir,
                      make_visuals=True):
    """Combine input CSVs, clean, consolidate attack classes, save.
    Returns the cleaned DataFrame.
    """
    print(f"\n{'='*70}")
    print(f"  CIC-IDS2017 preprocessing")
    print(f"{'='*70}")
    print(f"  Input folder: {input_folder}")
    print(f"  Output      : {os.path.join(output_dir, CICIDS_OUT_FILENAME)}")

    # ─── Combine input CSVs ────────────────────────────────────────────────
    print(f"\n  Combining input CSVs...")
    df = combine_cicids_csvs(input_folder)
    initial_shape = df.shape

    # ─── Normalize label column name ───────────────────────────────────────
    label_col = _normalize_cicids_label_column(df)

    # ─── Add binary label column ───────────────────────────────────────────
    # 0 = BENIGN, 1 = ATTACK. Created directly as int (no string phase).
    df['b_label'] = (df[label_col].astype(str).str.strip() != 'BENIGN').astype(int)
    print(f"    Added b_label column (0 = BENIGN, 1 = ATTACK)")

    # ─── Stage 1 EDA (before cleaning) ────────────────────────────────────
    if make_visuals:
        print(f"\n  Stage 1 EDA (before cleaning)...")
        mc_counts = df[label_col].astype(str).str.strip().value_counts()
        plot_distribution(
            mc_counts,
            xlabel="Attack Category", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage1_multiclass.png"),
        )
        bin_counts = df['b_label'].value_counts().sort_index()
        plot_distribution(
            bin_counts,
            xlabel="Label (0 = Benign, 1 = Attack)", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage1_binary.png"),
            color=['skyblue', 'lightcoral'],
        )

    # ─── Cleaning ──────────────────────────────────────────────────────────
    print(f"\n  Cleaning...")

    null_counts = df.isnull().sum()
    if null_counts.any():
        print(f"    Nulls in {(null_counts > 0).sum()} column(s); dropping rows.")
    df.dropna(inplace=True)
    print(f"    Shape after nulls: {df.shape}")

    # Infinite values (common in CIC-IDS flow-based features)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    inf_rows = df.isnull().any(axis=1).sum()
    print(f"    Rows with inf: {inf_rows}")
    df.dropna(inplace=True)
    print(f"    Shape after inf: {df.shape}")

    dupes = df.duplicated().sum()
    print(f"    Duplicate rows: {dupes}")
    df.drop_duplicates(inplace=True)
    print(f"    Shape after dupes: {df.shape}")

    df[label_col] = df[label_col].astype(str).str.strip()

    # ─── Consolidate attack classes ────────────────────────────────────────
    df[label_col] = df[label_col].replace(CICIDS_LABEL_MAP)
    print(f"\n    Class distribution after consolidation:")
    for cls, cnt in df[label_col].value_counts().items():
        print(f"      {cls:<20s} {cnt:>10,}")

    # ─── Rare-class removal ────────────────────────────────────────────────
    # Infiltration has only ~36 samples in raw CIC-IDS2017 — too few to
    # support faithful fold-wise resampling without generating overwhelmingly
    # synthetic training data for it.
    CICIDS_DROP_CLASSES = ['Infiltration']
    before_drop = len(df)
    df = df[~df[label_col].isin(CICIDS_DROP_CLASSES)].reset_index(drop=True)
    print(f"\n    Dropped classes {CICIDS_DROP_CLASSES}: "
          f"{before_drop - len(df)} rows removed")
    print(f"    Class distribution after drop:")
    for cls, cnt in df[label_col].value_counts().items():
        print(f"      {cls:<20s} {cnt:>10,}")
    print(f"\n    Total rows removed overall: {initial_shape[0] - df.shape[0]}")

    # ─── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, CICIDS_OUT_FILENAME)
    df.to_csv(out_path, index=False)
    print(f"    Saved: {out_path}")

    # ─── Stage 2 EDA (after cleaning) ──────────────────────────────────────
    if make_visuals:
        print(f"\n  Stage 2 EDA (after cleaning)...")
        mc_counts_clean = df[label_col].value_counts()
        plot_distribution(
            mc_counts_clean,
            xlabel="Attack Category", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage2_multiclass.png"),
        )
        bin_counts_clean = df['b_label'].value_counts().sort_index()
        plot_distribution(
            bin_counts_clean,
            xlabel="Label (0 = Benign, 1 = Attack)", ylabel="Count",
            filename=os.path.join(visuals_dir, "stage2_binary.png"),
            color=['skyblue', 'lightcoral'],
        )

    return df


##############################################################################
# ───────────────────────────────  CLI  ────────────────────────────────────
##############################################################################

def _resolve_paths(args):
    """Compute all input/output directories from the resolved arguments,
    creating output directories as needed.
    """
    unsw_out_dir   = os.path.join(args.output_base, UNSW_OUT_SUBDIR)
    cicids_out_dir = os.path.join(args.output_base, CICIDS_OUT_SUBDIR)
    unsw_vis_dir   = os.path.join(args.output_base, UNSW_VIS_SUBDIR)
    cicids_vis_dir = os.path.join(args.output_base, CICIDS_VIS_SUBDIR)

    for d in [unsw_out_dir, cicids_out_dir, unsw_vis_dir, cicids_vis_dir]:
        os.makedirs(d, exist_ok=True)

    return dict(
        unsw_out_dir=unsw_out_dir,
        cicids_out_dir=cicids_out_dir,
        unsw_vis_dir=unsw_vis_dir,
        cicids_vis_dir=cicids_vis_dir,
    )


def _output_exists(dirpath, filename):
    return os.path.exists(os.path.join(dirpath, filename))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="preprocess.py",
        description=(
            "Clean UNSW-NB15 and/or CIC-IDS2017 and write the cleaned CSVs "
            "consumed by ids_pipeline / MuDCANet.py. No scaling or "
            "resampling — those are handled per fold at training time."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", choices=["unsw", "cicids", "both"], default="both",
        help="Which dataset(s) to process.",
    )
    parser.add_argument(
        "--unsw-input", default=DEFAULT_UNSW_INPUT,
        help="Path to the raw UNSW-NB15 CSV.",
    )
    parser.add_argument(
        "--cicids-input", default=DEFAULT_CICIDS_INPUT,
        help="Path to the folder containing the raw CIC-IDS CSVs.",
    )
    parser.add_argument(
        "--output-base", default=DEFAULT_OUTPUT_BASE,
        help=(
            "Base output directory. Cleaned CSVs and visuals are written "
            "beneath this into the standard subfolders "
            f"({UNSW_OUT_SUBDIR}, {CICIDS_OUT_SUBDIR}, etc.)."
        ),
    )
    parser.add_argument(
        "--no-visuals", action="store_true",
        help="Skip stage-1 and stage-2 EDA plots.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing cleaned CSVs. Without this flag, "
             "datasets whose cleaned output already exists are skipped.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    paths = _resolve_paths(args)
    make_visuals = not args.no_visuals

    do_unsw   = args.dataset in ("unsw",   "both")
    do_cicids = args.dataset in ("cicids", "both")

    # ─── UNSW ──────────────────────────────────────────────────────────────
    if do_unsw:
        if (_output_exists(paths["unsw_out_dir"], UNSW_OUT_FILENAME)
                and not args.force):
            print(f"\n[UNSW] Cleaned output already exists at "
                  f"{os.path.join(paths['unsw_out_dir'], UNSW_OUT_FILENAME)}."
                  f"\n       Use --force to overwrite. Skipping.")
        else:
            preprocess_unsw(
                input_path=args.unsw_input,
                output_dir=paths["unsw_out_dir"],
                visuals_dir=paths["unsw_vis_dir"],
                make_visuals=make_visuals,
            )

    # ─── CIC-IDS ───────────────────────────────────────────────────────────
    if do_cicids:
        if (_output_exists(paths["cicids_out_dir"], CICIDS_OUT_FILENAME)
                and not args.force):
            print(f"\n[CIC-IDS] Cleaned output already exists at "
                  f"{os.path.join(paths['cicids_out_dir'], CICIDS_OUT_FILENAME)}."
                  f"\n          Use --force to overwrite. Skipping.")
        else:
            preprocess_cicids(
                input_folder=args.cicids_input,
                output_dir=paths["cicids_out_dir"],
                visuals_dir=paths["cicids_vis_dir"],
                make_visuals=make_visuals,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()