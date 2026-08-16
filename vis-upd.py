#!/usr/bin/env python3
##############################################################################
#####                                                                    #####
#####     VIS_MODULE_CHARTS.PY — STANDALONE                              #####
#####                                                                    #####
#####   Generates the connectivity and attention module comparison       #####
#####   charts. Runs independently of vis.py — reads the same saved      #####
#####   .npy prediction arrays and writes to the same output folder,     #####
#####   so it can be run before, after, or instead of vis.py without     #####
#####   interfering with it.                                            #####
#####                                                                    #####
#####   Figures produced (per dataset):                                  #####
#####     {ds}_connectivity_accuracy_bar.png                             #####
#####     {ds}_attention_accuracy_line.png                               #####
#####                                                                    #####
#####   The connectivity bars use the same glassy 3D treatment as the    #####
#####   stacked distribution bars in prep.py (gradient front face, top   #####
#####   and end faces for depth, specular highlight strip), adapted to   #####
#####   horizontal orientation — so both figures read as one visual      #####
#####   family in the paper.                                             #####
#####                                                                    #####
#####   Usage:                                                           #####
#####     python vis_module_charts.py                                    #####
#####                                                                    #####
##############################################################################

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Polygon
from matplotlib.colors import to_rgb, LinearSegmentedColormap
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")


##############################################################################
# ========================  CONFIGURATION  =================================
##############################################################################

_BASE    = "/mnt/c/Users/Battosai Himura/Desktop/Projects/tf-gpu-env/ids"
EXP_ROOT = os.path.join(_BASE, "experiment_results")

OUT_DIR = os.path.join(EXP_ROOT, "VIS-upd")
os.makedirs(OUT_DIR, exist_ok=True)

DPI      = 800
DATASETS = ["cicids", "unsw"]

# ── Model → experiment subfolder ──────────────────────────────────────────
FOLDERS = {
    "ResCNN":   "residual_only",
    "HWNet":    "highway",
    "SECNN":    "se_only",
    "CBAM":     "cbam",
    "Proposed": "MuDCANet",
}

# ── Colour palette — matches vis.py so figures stay consistent ────────────
COLOURS = {
    "ResCNN":   "#17becf",   # teal
    "HWNet":    "#8c564b",   # brown
    "SECNN":    "#bcbd22",   # olive
    "CBAM":     "#7f7f7f",   # grey
    "Proposed": "#e6a817",   # deep amber  ← novel contribution
}

# ── Display names shown on the figures ────────────────────────────────────
DISPLAY_NAMES = {
    "ResCNN":   "ResNet",
    "HWNet":    "HWNet",
    "SECNN":    "SE",
    "CBAM":     "CBAM",
    "Proposed": "Proposed",
}

# ── Model groups per chart ────────────────────────────────────────────────
CONNECTIVITY_MODELS = ["ResCNN", "HWNet", "Proposed"]
ATTENTION_MODELS    = ["SECNN", "CBAM", "Proposed"]

# Which model gets the emphasis treatment
HERO_MODEL = "Proposed"


##############################################################################
# ========================  STYLE CONSTANTS  ===============================
##############################################################################

plt.rcParams["font.weight"]      = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["font.size"]        = 12.5
plt.rcParams["axes.labelsize"]   = 12.5
plt.rcParams["xtick.labelsize"]  = 12.5
plt.rcParams["ytick.labelsize"]  = 12.5
plt.rcParams["legend.fontsize"]  = 12.5

TRACK_COLOUR   = "#E4E8ED"   # recessed channel behind each bar
BASELINE_ALPHA = 0.80        # baselines slightly muted
HERO_ALPHA     = 1.00        # proposed model at full saturation
TEXT_DARK      = "#1A1A2E"
TEXT_MUTED     = "#6B7280"
GRID_COLOUR    = "#D9DEE3"
HIGHLIGHT_COL  = "#FFF4D6"   # column behind the hero model

TASK_COLOURS = {"binary": "#d62728", "multiclass": "#1f77b4"}
TASK_MARKERS = {"binary": "^",       "multiclass": "s"}

# ── 3D bar geometry (connectivity chart) ──────────────────────────────────
# Bar face height in y-units. Bars sit 1.0 apart, so the gap between
# adjacent bars is (1.0 - BAR_H). At 0.59 the gap is 0.41 — roughly
# two-thirds of the previous 0.62 gap.
BAR_H = 0.59

# Depth offset. DEPTH_Y is in y-units; DEPTH_X_FRAC is a fraction of the
# x-axis range, so the extrusion stays visually square regardless of how
# wide the accuracy window happens to be.
DEPTH_Y      = 0.13
DEPTH_X_FRAC = 0.022


##############################################################################
# ========================  DATA LOADING  ==================================
##############################################################################

def npy_path(model, task, ds, array_type):
    """Full path to a saved prediction array."""
    return os.path.join(EXP_ROOT, FOLDERS[model], "npy", task,
                        f"{model}_{task}_{ds}_{array_type}.npy")


def load_accuracy(model, task, ds):
    """Load y_true/y_pred and return accuracy as a percentage.
    Returns None if either array is missing.
    """
    p_true = npy_path(model, task, ds, "y_true")
    p_pred = npy_path(model, task, ds, "y_pred")
    if not os.path.exists(p_true) or not os.path.exists(p_pred):
        print(f"  [WARN] Missing arrays for {model} | {task} | {ds}")
        return None
    y_true = np.load(p_true, allow_pickle=True)
    y_pred = np.load(p_pred, allow_pickle=True)
    return accuracy_score(y_true, y_pred) * 100


def out_path(filename):
    return os.path.join(OUT_DIR, filename)


##############################################################################
# ========  GLASSY 3D BAR PRIMITIVES  ======================================
##############################################################################
# Same approach as prep.py's draw_3d_segment(), rotated to horizontal:
# a gradient-filled front face, flat top face and gradient end face for
# depth, plus a specular highlight strip for the glass look.
##############################################################################

def _shade(hex_colour, factor):
    """Lighten (factor > 1) or darken (factor < 1) an RGB colour."""
    r, g, b = to_rgb(hex_colour)
    if factor < 1:
        return (r * factor, g * factor, b * factor)
    f = factor - 1
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _gradient_fill(ax, x0, x1, y0, y1, c_top, c_bottom,
                   zorder, clip_poly, alpha=1.0):
    """Vertical gradient inside a bbox, clipped to a polygon."""
    if x1 <= x0 or y1 <= y0:
        return None
    cmap = LinearSegmentedColormap.from_list("g", [c_bottom, c_top])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    im = ax.imshow(grad, extent=[x0, x1, y0, y1], origin="lower",
                   aspect="auto", cmap=cmap, zorder=zorder, alpha=alpha,
                   interpolation="bilinear")
    im.set_clip_path(clip_poly)
    return im


def _draw_3d_hbar(ax, x_left, x_right, y_center, height, base_colour,
                  dx, dy, zbase, alpha=1.0, gloss=True, edge_lw=1.0):
    """Glassy 3D horizontal bar.

    Faces drawn:
      front  — gradient (lighter top → saturated bottom)
      top    — flat lighter fill, offset by (dx, dy)
      end    — gradient (darker) on the right cap, giving the extrusion
      gloss  — white specular strip across the upper front face
    """
    y0 = y_center - height / 2.0
    y1 = y_center + height / 2.0
    edge = _shade(base_colour, 0.45)

    # ── FRONT face ────────────────────────────────────────────────────────
    front = Polygon(
        [(x_left, y0), (x_right, y0), (x_right, y1), (x_left, y1)],
        closed=True, facecolor="none", edgecolor=edge,
        linewidth=edge_lw, zorder=zbase + 1, alpha=alpha)
    ax.add_patch(front)
    _gradient_fill(ax, x_left, x_right, y0, y1,
                   c_top=_shade(base_colour, 1.42),
                   c_bottom=_shade(base_colour, 0.88),
                   zorder=zbase, clip_poly=front, alpha=alpha)

    # ── TOP face ──────────────────────────────────────────────────────────
    top = Polygon(
        [(x_left, y1), (x_right, y1),
         (x_right + dx, y1 + dy), (x_left + dx, y1 + dy)],
        closed=True, facecolor=_shade(base_colour, 1.30),
        edgecolor=edge, linewidth=edge_lw, zorder=zbase + 2, alpha=alpha)
    ax.add_patch(top)

    # ── RIGHT end face ────────────────────────────────────────────────────
    end = Polygon(
        [(x_right, y0), (x_right + dx, y0 + dy),
         (x_right + dx, y1 + dy), (x_right, y1)],
        closed=True, facecolor="none", edgecolor=edge,
        linewidth=edge_lw, zorder=zbase + 2, alpha=alpha)
    ax.add_patch(end)
    _gradient_fill(ax, x_right, x_right + dx, y0, y1 + dy,
                   c_top=_shade(base_colour, 0.95),
                   c_bottom=_shade(base_colour, 0.58),
                   zorder=zbase + 1, clip_poly=end, alpha=alpha)

    # ── SPECULAR HIGHLIGHT on the front face ──────────────────────────────
    if gloss:
        hl_y0 = y0 + height * 0.60
        hl_y1 = y0 + height * 0.85
        hl = Polygon(
            [(x_left, hl_y0), (x_right, hl_y0),
             (x_right, hl_y1), (x_left, hl_y1)],
            closed=True, facecolor="none", edgecolor="none",
            zorder=zbase + 3)
        ax.add_patch(hl)
        _gradient_fill(ax, x_left, x_right, hl_y0, hl_y1,
                       c_top=(1, 1, 1), c_bottom=(1, 1, 1),
                       zorder=zbase + 3, clip_poly=hl, alpha=0.20 * alpha)


##############################################################################
# ========  CONNECTIVITY MODULE 3D HORIZONTAL BAR CHART  ===================
##############################################################################

def plot_connectivity_accuracy_bar(ds):
    """
    Glassy 3D horizontal accuracy bar chart for connection module
    comparison. Models: ResNet, HWNet, MuDCANet.

    Design:
      - Extruded 3D bars with gradient front face, top and end faces,
        and a specular highlight strip — matching the visual treatment
        of the distribution bars in prep.py.
      - Each bar sits in a recessed grey channel spanning the full axis,
        so the bar reads as filling a container and the truncated
        x-window stays visible rather than hidden.
      - Emphasis hierarchy: baselines slightly muted, proposed model at
        full saturation with a bolder tick label.
      - Delta badge on the proposed bar showing gain over the best
        baseline in percentage points.
      - Sparse visible x-axis so the zoom level is honest.

    Binary and multiclass shown as side-by-side subplots. Subplot titles
    are intentionally omitted — label the panels externally.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))

    for ax, task in zip(axes, ["binary", "multiclass"]):
        accs, labels, cols = [], [], []
        for model in CONNECTIVITY_MODELS:
            acc = load_accuracy(model, task, ds)
            if acc is not None:
                accs.append(acc)
                labels.append(model)
                cols.append(COLOURS[model])

        if not accs:
            ax.set_visible(False)
            continue

        # ── Axis window ───────────────────────────────────────────────────
        # Truncated window, padded from the observed spread so the
        # shortest bar still has visible length and labels have room.
        span  = max(max(accs) - min(accs), 0.05)
        pad_l = max(span * 0.8, 0.35)
        pad_r = max(span * 2.6, 1.30)      # room for depth + value labels
        x_min = min(accs) - pad_l
        x_max = max(accs) + pad_r

        dx = (x_max - x_min) * DEPTH_X_FRAC
        dy = DEPTH_Y

        y_pos = np.arange(len(labels))

        # ── Recessed channel behind each bar ──────────────────────────────
        # Drawn without gloss so it reads as a groove, not a second bar.
        for y in y_pos:
            _draw_3d_hbar(ax, x_min, x_max - dx, y, BAR_H,
                          TRACK_COLOUR, dx, dy,
                          zbase=1, alpha=1.0, gloss=False, edge_lw=0.7)

        # ── Bars ──────────────────────────────────────────────────────────
        for y, val, model, col in zip(y_pos, accs, labels, cols):
            is_hero = (model == HERO_MODEL)
            _draw_3d_hbar(ax, x_min, val, y, BAR_H, col, dx, dy,
                          zbase=10,
                          alpha=HERO_ALPHA if is_hero else BASELINE_ALPHA,
                          gloss=True, edge_lw=1.0)

        # ── Dashed reference line at best baseline ────────────────────────
        baseline_accs = [a for a, m in zip(accs, labels) if m != HERO_MODEL]
        best_baseline = max(baseline_accs) if baseline_accs else None
        if best_baseline is not None:
            ax.axvline(best_baseline, color="#B8C0CA", lw=1.2,
                       linestyle="--", zorder=9)

        # ── Value labels, clear of the extruded end face ──────────────────
        label_gap = dx + span * 0.10 + 0.05
        for y, val, model in zip(y_pos, accs, labels):
            is_hero = (model == HERO_MODEL)
            ax.text(val + label_gap, y + dy / 2, f"{val:.2f}%",
                    va="center", ha="left",
                    fontsize=13.5 if is_hero else 12.0,
                    fontweight="bold",
                    color=TEXT_DARK if is_hero else TEXT_MUTED,
                    zorder=20)

        # ── Delta badge on the hero bar ───────────────────────────────────
        if best_baseline is not None and HERO_MODEL in labels:
            hero_i   = labels.index(HERO_MODEL)
            hero_val = accs[hero_i]
            delta    = hero_val - best_baseline
            if abs(delta) > 1e-9:
                sign = "+" if delta > 0 else ""
                ax.text(hero_val + label_gap,
                        y_pos[hero_i] + dy / 2 - 0.30,
                        f"{sign}{delta:.2f} pp vs. best baseline",
                        va="center", ha="left",
                        fontsize=9.5, fontweight="bold",
                        color="#1a7a3a" if delta > 0 else "#b02a2a",
                        zorder=20)

        # ── Axis cosmetics ────────────────────────────────────────────────
        # Limits are set AFTER all drawing: the gradient fills use imshow,
        # which would otherwise rescale the axes.
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([-0.60, len(labels) - 1 + 0.60 + dy])

        ax.set_yticks(y_pos)
        ax.set_yticklabels([DISPLAY_NAMES[m] for m in labels],
                           fontsize=12.5, fontweight="bold")
        for tick_label, model in zip(ax.get_yticklabels(), labels):
            if model == HERO_MODEL:
                tick_label.set_fontsize(13.5)
                tick_label.set_color(TEXT_DARK)
            else:
                tick_label.set_color(TEXT_MUTED)

        # Sparse x-axis — enough to show the zoom level without clutter
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.tick_params(axis="x", labelsize=10.5, colors=TEXT_MUTED,
                       length=0, pad=6)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.set_xlabel("Accuracy (%)", fontsize=12.0, fontweight="bold",
                      color=TEXT_MUTED, labelpad=8)

        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_visible(False)

    plt.tight_layout(w_pad=3.0)
    fname = f"{ds}_connectivity_accuracy_bar.png"
    plt.savefig(out_path(fname), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


##############################################################################
# ========  ATTENTION VARIANT ACCURACY LINE CHART  =========================
##############################################################################

def plot_attention_accuracy_line(ds):
    """
    Accuracy line chart for attention mechanism comparison.
    Models: SE, CBAM, MuDCANet.

    Design:
      - Every point labelled with a white pill background, so exact
        values are readable without cross-referencing the axis.
      - Shaded band between the binary and multiclass lines turns the
        task gap into a visible area rather than a mental subtraction.
      - Highlighted column behind the proposed model so the eye lands
        on the contribution.
      - Larger markers with white halos, soft grid, frameless legend
        placed above the axes so it never overlaps data.

    The y-window is derived entirely from the observed values on both
    series, so every point is guaranteed visible regardless of how far
    apart the baselines and the proposed model sit.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x_pos   = np.arange(len(ATTENTION_MODELS))

    # ── Gather all values first (needed for band + y-limits) ──────────────
    series = {}
    for task in ["binary", "multiclass"]:
        accs = []
        for model in ATTENTION_MODELS:
            acc = load_accuracy(model, task, ds)
            accs.append(acc if acc is not None else np.nan)
        if not np.all(np.isnan(accs)):
            series[task] = np.array(accs, dtype=float)

    if not series:
        plt.close()
        print(f"  [WARN] No attention data for {ds}")
        return

    task_list = list(series.keys())

    # ── Highlight column behind the proposed model ────────────────────────
    if HERO_MODEL in ATTENTION_MODELS:
        hero_x = ATTENTION_MODELS.index(HERO_MODEL)
        ax.axvspan(hero_x - 0.35, hero_x + 0.35,
                   color=HIGHLIGHT_COL, alpha=0.75, zorder=0)

    # ── Shaded band between the two task lines ────────────────────────────
    if len(series) == 2:
        lo = np.minimum(series[task_list[0]], series[task_list[1]])
        hi = np.maximum(series[task_list[0]], series[task_list[1]])
        ax.fill_between(x_pos, lo, hi, color="#8899AA", alpha=0.13,
                        zorder=1, linewidth=0)

    # ── Lines + markers ───────────────────────────────────────────────────
    for task, accs in series.items():
        ax.plot(x_pos, accs, color=TASK_COLOURS[task], lw=2.6,
                marker=TASK_MARKERS[task], markersize=11,
                label=task.capitalize(),
                markerfacecolor=TASK_COLOURS[task],
                markeredgecolor="white", markeredgewidth=1.6,
                zorder=3, solid_capstyle="round")

    # ── Label every point ─────────────────────────────────────────────────
    # Offset direction chosen per point so the two series don't collide;
    # exact ties are broken deterministically by series order.
    for ti, (task, accs) in enumerate(series.items()):
        col = TASK_COLOURS[task]
        for xi, val in zip(x_pos, accs):
            if np.isnan(val):
                continue
            if len(series) == 2:
                other_val = series[task_list[1 - ti]][xi]
                if np.isnan(other_val):
                    above = True
                elif abs(val - other_val) < 1e-9:
                    above = (ti == 0)          # deterministic tie-break
                else:
                    above = val > other_val
            else:
                above = True
            ax.annotate(
                f"{val:.2f}%",
                xy=(xi, val),
                xytext=(0, 14 if above else -18),
                textcoords="offset points",
                fontsize=11.0, fontweight="bold", color=col,
                ha="center", va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                          edgecolor="none", alpha=0.82))

    # ── Axis cosmetics ────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels([DISPLAY_NAMES[m] for m in ATTENTION_MODELS],
                       fontsize=13.5, fontweight="bold")
    for tick_label, model in zip(ax.get_xticklabels(), ATTENTION_MODELS):
        tick_label.set_color(TEXT_DARK if model == HERO_MODEL else TEXT_MUTED)

    ax.set_xlim([-0.45, len(ATTENTION_MODELS) - 0.55])

    # ── Y-window derived from the data ────────────────────────────────────
    # Padding is proportional to the observed spread, so the window adapts
    # whether the models are tightly clustered or several points apart.
    # Every plotted value is inside the window by construction.
    all_vals = np.concatenate([v[~np.isnan(v)] for v in series.values()])
    v_min, v_max = float(all_vals.min()), float(all_vals.max())
    spread = max(v_max - v_min, 0.5)      # guard against a flat series
    y_lo = max(0.0,   v_min - spread * 0.28)
    y_hi = min(100.0, v_max + spread * 0.28)
    ax.set_ylim([y_lo, y_hi])
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))

    ax.set_ylabel("Accuracy", fontsize=13.5, fontweight="bold",
                  color=TEXT_DARK, labelpad=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.30, color=GRID_COLOUR)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=11.5, colors=TEXT_MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=8)

    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.12),
              mode="expand", ncol=2, frameon=False,
              prop={"weight": "bold", "size": 12.5})

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOUR)

    plt.tight_layout()
    fname = f"{ds}_attention_accuracy_line.png"
    plt.savefig(out_path(fname), dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


##############################################################################
# ========================  MAIN  ==========================================
##############################################################################

def main():
    print("\n" + "=" * 65)
    print("  VIS_MODULE_CHARTS.PY — connectivity + attention figures")
    print("=" * 65)

    print("\n[1] Connectivity module accuracy bar charts (3D)...")
    for ds in DATASETS:
        print(f"  Dataset: {ds}")
        plot_connectivity_accuracy_bar(ds)

    print("\n[2] Attention mechanism accuracy line charts...")
    for ds in DATASETS:
        print(f"  Dataset: {ds}")
        plot_attention_accuracy_line(ds)

    print("\n" + "=" * 65)
    print("  COMPLETE")
    print(f"  Figures → {OUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()