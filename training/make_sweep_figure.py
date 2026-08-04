"""Render the support-size sweep as a paper figure (PDF + PNG).

Main figure: macro R^2 vs support size (log-2 x-axis), one line per
support-using method with a +-1 std band across seeds, gray dashed/dotted
reference lines for the zero-support baselines, and a labeled R^2 = 0 line
("predicting regional mean"). Sized for a single column.

Faceted figure: the same content per held-out region on a 2x3 grid (free y),
where Basin & Range (n_query=35) is expected to be the informative facet.

Colors are the validated categorical palette (fixed slot order); the three
hues below 3:1 contrast on a light surface carry direct end-of-line labels
per the relief rule, with marker shape as secondary encoding.

Usage:
    python training/make_sweep_figure.py --raw results/sweep/sweep_raw.csv \
        --out results/figures/ [--suffix _monthly]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SERIES = [  # (method, hex, marker, direct_label)
    ("ridge_support_only", "#2a78d6", "o", False),
    ("ridge_refit",        "#eb6834", "s", False),
    ("mlp_finetune",       "#1baf7a", "^", True),
    ("ccpa_no_climate",    "#eda100", "D", True),
    ("ccpa",               "#e87ba4", "v", True),
]
REFS = [("ridge", "#52514e", (0, (5, 2.5))), ("mlp", "#8a8984", (0, (1.5, 2)))]
ZERO_COLOR = "#b9b8b3"
GRID_COLOR = "#e8e8e6"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SIZES = [8, 16, 32, 64]

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
    "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.7,
    "text.color": TEXT_PRIMARY, "axes.labelcolor": TEXT_PRIMARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "pdf.fonttype": 42,
})


def macro_stats(raw):
    """Per-seed macro (equal-weight over regions), then mean/std across seeds."""
    per_seed = raw.groupby(["seed", "method", "support_size"])["r2"] \
                  .mean().reset_index()
    return per_seed.groupby(["method", "support_size"])["r2"] \
                   .agg(["mean", "std"]).reset_index()


def region_stats(raw):
    return raw.groupby(["held_out_region", "method", "support_size"])["r2"] \
              .agg(["mean", "std"]).reset_index()


def style_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.tick_params(which="minor", bottom=False)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ridge_refit and the zero-support references run so far negative on monthly
# features that including them would compress every other series into a sliver.
# The y-range is set from the informative methods and anything outside is named
# in a footnote rather than silently cropped.
FOCUS = ["ridge_support_only", "mlp_finetune", "ccpa_no_climate", "ccpa"]


def focus_ylim(stats, refs, pad=0.12):
    f = stats[stats["method"].isin(FOCUS)]
    lo = float((f["mean"] - f["std"].fillna(0)).min())
    hi = float((f["mean"] + f["std"].fillna(0)).max())
    hi = max(hi, 0.02)  # always show the R^2 = 0 line
    span = hi - lo
    return lo - pad * span, hi + pad * span


def offscale_note(ax, stats, refs, lo):
    """Name any series whose whole curve sits below the visible range."""
    names = []
    for method, _, _, _ in SERIES:
        if method in FOCUS:
            continue
        s = stats[stats["method"] == method]
        if not s.empty and float(s["mean"].max()) < lo:
            names.append(method)
    names += [m for m, v in refs.items() if v < lo]
    if names:
        ax.annotate("below axis: " + ", ".join(names),
                    xy=(0.5, 0.012), xycoords="axes fraction", ha="center",
                    fontsize=5.8, color=TEXT_SECONDARY)


def draw_panel(ax, stats, refs, zero_label=False, direct_labels=False,
               band=True, lw=1.6, ms=3.6):
    ax.axhline(0, color=ZERO_COLOR, linewidth=0.8, zorder=1)
    if zero_label:
        ax.annotate("predicting regional mean", xy=(SIZES[0], 0),
                    xytext=(0, 2.5), textcoords="offset points",
                    fontsize=6, style="italic", color=TEXT_SECONDARY)
    for method, color, dash in REFS:
        if method in refs:
            ax.axhline(refs[method], color=color, linewidth=0.9,
                       linestyle=dash, zorder=2)
    ends = []
    for method, color, marker, label_it in SERIES:
        s = stats[stats["method"] == method].sort_values("support_size")
        if s.empty:
            continue
        ax.plot(s["support_size"], s["mean"], color=color, marker=marker,
                markersize=ms, linewidth=lw, zorder=4,
                markeredgecolor="white", markeredgewidth=0.5)
        if band and s["std"].notna().all():
            ax.fill_between(s["support_size"], s["mean"] - s["std"],
                            s["mean"] + s["std"], color=color, alpha=0.15,
                            linewidth=0, zorder=3)
        if direct_labels and label_it:
            ends.append((float(s["mean"].iloc[-1]), method, color))
    return ends


def place_end_labels(ax, ends, x=SIZES[-1]):
    """Right-edge direct labels, nudged apart to avoid collisions."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.095  # ~1 label height at this figure size
    ends = sorted(ends)
    ys = [ends[0][0]]
    for y, _, _ in ends[1:]:
        ys.append(max(y, ys[-1] + gap))
    for (y0, method, color), y in zip(ends, ys):
        # Anchor at the nudged y (not y0) so the collision spacing is what
        # actually gets drawn; a leader line ties it back to the data point.
        ax.annotate(method, xy=(x, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=6.5,
                    color=TEXT_PRIMARY,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=color, lw=0.7))
        if abs(y - y0) > 1e-9:
            ax.plot([x, x], [y0, y], color=color, linewidth=0.6,
                    alpha=0.55, zorder=3)


def legend_handles(refs_present):
    handles = [plt.Line2D([], [], color=c, marker=m, markersize=3.6,
                          linewidth=1.6, markeredgecolor="white",
                          markeredgewidth=0.5, label=meth)
               for meth, c, m, _ in SERIES]
    handles += [plt.Line2D([], [], color=c, linewidth=0.9, linestyle=d,
                           label="%s (no support)" % meth)
                for meth, c, d in REFS if meth in refs_present]
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/sweep/sweep_raw.csv")
    ap.add_argument("--out", default="results/figures/")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_seeds = raw["seed"].nunique()

    swept = raw[raw["support_size"] > 0]
    ref_macro = raw[raw["support_size"] == 0] \
        .groupby(["seed", "method"])["r2"].mean() \
        .groupby("method").mean().to_dict()

    # --- Main single-column figure -------------------------------------
    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
    stats = macro_stats(swept)
    ends = draw_panel(ax, stats, ref_macro, zero_label=True, direct_labels=True)
    style_axis(ax)
    ax.set_xlabel("Support size (labeled samples from held-out region)")
    ax.set_ylabel("Macro R² (mean ± 1 std, %d seeds)" % n_seeds)
    ax.set_xlim(SIZES[0] * 0.9, SIZES[-1] * 1.6)  # room for end labels
    lo, hi = focus_ylim(stats, ref_macro)
    ax.set_ylim(lo, hi)
    offscale_note(ax, stats, ref_macro, lo)
    place_end_labels(ax, ends)
    ax.legend(handles=legend_handles(ref_macro), loc="lower right",
              frameon=False, handlelength=1.8, labelspacing=0.35)
    for ext in ("pdf", "png"):
        fig.savefig(out / ("support_sweep%s.%s" % (args.suffix, ext)), dpi=300)
    plt.close(fig)

    # --- Per-region 2x3 facet grid -------------------------------------
    regions = (raw.groupby("held_out_region")["n_query"].first()
               .sort_values(ascending=False))
    ref_region = raw[raw["support_size"] == 0] \
        .groupby(["held_out_region", "method"])["r2"].mean()
    rstats = region_stats(swept)

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.4), constrained_layout=True)
    for ax, (region, nq) in zip(axes.ravel(), regions.items()):
        refs = ref_region.loc[region].to_dict()
        rs = rstats[rstats["held_out_region"] == region]
        draw_panel(ax, rs, refs, lw=1.3, ms=3.0)
        style_axis(ax)
        lo, hi = focus_ylim(rs, refs)
        ax.set_ylim(lo, hi)
        offscale_note(ax, rs, refs, lo)
        flag = ", noisy" if nq < 100 else ""
        ax.set_title("%s  (query n=%d%s)" % (region, nq, flag), fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("Support size")
    for ax in axes[:, 0]:
        ax.set_ylabel("R²")
    fig.legend(handles=legend_handles(ref_macro), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    for ext in ("pdf", "png"):
        fig.savefig(out / ("support_sweep_by_region%s.%s" % (args.suffix, ext)),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote 4 files to %s (suffix '%s')" % (out, args.suffix))


if __name__ == "__main__":
    main()
