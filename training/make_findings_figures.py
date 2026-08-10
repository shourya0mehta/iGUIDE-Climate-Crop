"""Figures for results/FINDINGS.md (single-column, publication style).

1. partition_method_heatmap : macro R^2, method x partition, both feature
   sets side by side. Annotated cells; diverging color (R^2 polarity around
   0), clipped so the extreme negatives don't flatten the scale — the number
   in the cell is the datum, color is a reading aid.
2. shift_vs_transfer : the three shift measures vs mlp_finetune R^2 per
   (partition, held-out region) cell (seed-averaged, monthly features),
   OLS fit with a bootstrap 95% CI band. Points colored by partition
   (first three categorical slots — validated all-pairs).
3. coef_cosine_heatmap : pairwise cosine similarity of per-region RidgeCV
   coefficient vectors, ERS partition (diverging around 0).

The updated support-sweep figure is produced by the existing
training/make_sweep_figure.py on the RidgeCV-spliced sweep raws.

Usage:
    python training/make_findings_figures.py --out results/figures/
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.loco_eval import METHOD_ORDER  # noqa: E402
from training.partition_analysis import (  # noqa: E402
    FEATURE_SETS, PARTITIONS, PREDICTORS, RESPONSE_METHOD, load_runs,
    macro_table)

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e8e8e6"
PARTITION_COLORS = {"state": "#2a78d6", "ers": "#eb6834", "clim": "#1baf7a"}
PARTITION_LABELS = {"state": "state-level FRR", "ers": "county ERS",
                    "clim": "climate clusters"}
# diverging blue <-> red around a neutral gray midpoint (palette pair)
DIVERGING = LinearSegmentedColormap.from_list(
    "div", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.7,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42,
})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / ("%s.%s" % (name, ext)), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote %s/%s.{pdf,png}" % (out, name))


def annotate_cells(ax, M, norm, cmap, fmt="%+.2f", fontsize=6.2):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            r, g, b, _ = cmap(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, fmt % v, ha="center", va="center",
                    fontsize=fontsize,
                    color="white" if lum < 0.45 else TEXT_PRIMARY)


def fig_partition_heatmap(mac, out):
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.6), constrained_layout=True)
    for ax, f in zip(axes, FEATURE_SETS):
        piv = (mac[mac["features"] == f]
               .pivot(index="method", columns="partition", values="mean")
               .reindex(METHOD_ORDER)[PARTITIONS])
        M = np.clip(piv.values, -1.0, 0.5)
        ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")
        annotate_cells(ax, piv.values, norm, DIVERGING)
        ax.set_xticks(range(len(PARTITIONS)))
        ax.set_xticklabels([PARTITION_LABELS[p].replace(" ", "\n")
                            for p in PARTITIONS], fontsize=6.2)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels(METHOD_ORDER if ax is axes[0] else [])
        ax.set_title("%s (%d features)"
                     % (f, 5 if f == "seasonal" else 35), fontsize=7.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle("Macro R² by method × spatial partition", fontsize=8, y=1.04)
    axes[0].annotate("cell color clipped to [−1, +0.5]; printed value is exact",
                     xy=(0, -0.16), xycoords="axes fraction", fontsize=5.6,
                     color=TEXT_SECONDARY)
    save(fig, out, "partition_method_heatmap")


def boot_band(x, y, xs, reps=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    lines = np.empty((reps, len(xs)))
    for r in range(reps):
        idx = rng.integers(0, n, n)
        if np.ptp(x[idx]) == 0:
            lines[r] = np.nan
            continue
        b, a = np.polyfit(x[idx], y[idx], 1)
        lines[r] = a + b * xs
    return (np.nanpercentile(lines, 2.5, axis=0),
            np.nanpercentile(lines, 97.5, axis=0))


def fig_shift_scatter(cells, out):
    labels = {"energy_dist": "Energy distance (covariate shift)",
              "wass_yield": "Wasserstein, yield (bu/acre; label shift)",
              "inregion_r2": "In-region CV R² (intrinsic ceiling)"}
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.4), constrained_layout=True)
    y = cells["r2"].values
    for ax, pred in zip(axes, PREDICTORS):
        x = cells[pred].values
        xs = np.linspace(x.min(), x.max(), 100)
        b, a = np.polyfit(x, y, 1)
        lo, hi = boot_band(x, y, xs)
        ax.fill_between(xs, lo, hi, color="#b9b8b3", alpha=0.35, linewidth=0)
        ax.plot(xs, a + b * xs, color=TEXT_SECONDARY, linewidth=1.2)
        for p in PARTITIONS:
            m = (cells["partition"] == p).values
            ax.scatter(x[m], y[m], s=16, color=PARTITION_COLORS[p],
                       edgecolor="white", linewidth=0.5, zorder=4,
                       label=PARTITION_LABELS[p])
        ax.axhline(0, color="#b9b8b3", linewidth=0.8, zorder=1)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(labels[pred])
        ax.set_ylabel("%s R²" % RESPONSE_METHOD)
        r = np.corrcoef(x, y)[0, 1]
        ax.annotate("r = %+.2f" % r, xy=(0.98, 0.06), xycoords="axes fraction",
                    ha="right", fontsize=7, color=TEXT_PRIMARY)
    axes[0].legend(frameon=False, loc="lower right", handletextpad=0.1,
                   borderaxespad=0.2)
    fig.suptitle("Which shift predicts transfer failure?\n"
                 "(one point = one held-out fold, seed-averaged, "
                 "monthly features)", fontsize=8)
    save(fig, out, "shift_vs_transfer")


def fig_coef_cosine(out):
    mat = pd.read_csv(REPO_ROOT / "results/shift/coef_cosine_ers.csv",
                      index_col=0)
    order = mat.index.tolist()
    M = mat.values
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(3.5, 3.2), constrained_layout=True)
    ax.imshow(M, cmap=DIVERGING, norm=norm)
    annotate_cells(ax, M, norm, DIVERGING, fontsize=5.8)
    short = {"Heartland": "Heartland", "Northern Crescent": "N.Crescent",
             "Southern Seaboard": "S.Seaboard", "Prairie Gateway": "Pr.Gateway",
             "Eastern Uplands": "E.Uplands", "Mississippi Portal": "Miss.Portal",
             "Northern Great Plains": "N.Gt.Plains", "Fruitful Rim": "Fr.Rim"}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([short.get(r, r) for r in order], rotation=45,
                       ha="right", fontsize=6.2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short.get(r, r) for r in order], fontsize=6.2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Cosine similarity of per-region RidgeCV coefficients\n"
                 "(county-level ERS regions, 35 monthly features)", fontsize=7.5)
    save(fig, out, "coef_cosine_heatmap")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/figures/")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = load_runs()
    mac = macro_table(raw)
    fig_partition_heatmap(mac, out)

    cells = pd.read_csv(REPO_ROOT / "results/shift/cells_monthly.csv")
    fig_shift_scatter(cells, out)
    fig_coef_cosine(out)


if __name__ == "__main__":
    main()
