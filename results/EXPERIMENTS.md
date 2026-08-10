# Monthly features + support-size sweep

> **SUPERSEDED IN PART (2026-08-10) — read results/FINDINGS.md first.**
> Every `ridge_support_only` and `ridge_refit` number in this file was
> computed with a fixed `Ridge(alpha=1.0)`, which at p=35 features and
> n=8–32 support rows is far too weak a penalty; the local-only baseline
> numbers here are artifacts of under-regularization, not evidence about
> transfer. With the penalty chosen by internal leave-one-out CV
> (`RidgeCV`, results/sweep_monthly_cv/, results/partitions/),
> `ridge_support_only` at s=32 on monthly features moves from −0.161 to
> **+0.009** and wins at every support size, so **the headline claim below
> ("transfer wins when local labels are scarce, crossover around 32–64")
> does not survive**. The fixed-alpha numbers are retained below and at
> results/monthly/, results/sweep/, results/sweep_monthly/ as a robustness
> reference only. Non-ridge numbers (mlp*, ccpa*) are unaffected. The
> state-level `region` partition used throughout this file assigns whole
> states to Farm Resource Regions; the county-level correction and the
> partition-sensitivity experiment are in results/FINDINGS.md.

Both experiments reuse the corrected LOCO harness unchanged: same protocol, same
leakage guards (all active and passing), same method definitions, same
hyperparameters. Only the feature set (Exp 1) and support size (Exp 2) vary.
5 seeds x 6 folds throughout; CPU only; reruns byte-identical.

Datasets: `master_dataset.csv` (5 season-long features) and
`master_dataset_monthly.csv` (35 = 5 variables x 7 months, April-October).

---

## Headline answers

**(a) Do monthly features rescue transfer? Partly — and not the way it looks.**
Every transfer method improved substantially, but `ridge_support_only` got
*worse*, and roughly half the closed gap comes from that degradation rather than
from better transfer.

**(b) Does CCPA's advantage widen as support shrinks? On monthly features the
macro numbers say yes — but the effect is one region, not a general law.** The
`ccpa` - `mlp_finetune` macro gap grows monotonically as labels get scarcer
(+0.037 at s=64, +0.069 at s=32, +0.142 at s=16, +0.196 at s=8). That is
exactly the predicted shape. However, excluding Basin & Range (n_query=35,
flagged noisy) the sign flips and `ccpa` loses at every support size. The
trend is real but it is carried by the single smallest, most climatically
distinct region. On 5 features there is no such trend at all.

**(c) Does anything beat `ridge_support_only`? Yes — decisively, once labels are
scarce.** On monthly features at s=32, `mlp_finetune` (-0.139) and `ccpa`
(-0.149) edge past it (-0.161) in the main run. The sweep makes the effect much
larger: at s=8, `ccpa` (-0.268) and `mlp_finetune` (-0.465) both crush
`ridge_support_only` (-0.814), and the local-only baseline only retakes the
lead at s=64 (+0.118). **The strongest defensible claim in this work is the
label-efficiency one: transfer wins when local labels are scarce, and the
crossover is around 32-64 samples.**

The paper's original central claim — CCPA as *the* architectural fix for
geographic transfer — is still not supported. `ccpa` and `mlp_finetune` are
statistically indistinguishable in aggregate; a meta-learned initialization is
not buying anything a conventionally trained MLP with the same 10-step
adaptation does not already provide, except on Basin & Range.

---

## Experiment 1 — monthly climate aggregates

`data/build_master_monthly.py` rebuilds the dataset per (county, month) instead
of per county-season, pivoted wide to 35 features named `{variable}_{MM}`. Same
source CSVs, same growing season, same sum-vs-mean treatment. All acceptance
checks passed: 8691 rows x 42 cols, zero nulls, exact region counts, Champaign
IL 2019 = 180.8, and fips/year key-set identical to `master_dataset.csv`.

Seasonal-vs-monthly reconciliation (mean / max absolute difference), reported
not asserted:

| variable | mean diff | max diff |
|---|---|---|
| avg_temp | 1.3e-2 K | 3.9e-2 K |
| total_precip | 7.9e-13 | 1.5e-11 |
| avg_humidity | 2.9e-2 % | 1.0e-1 % |
| avg_radiation | 2.9 W/m^2 | 11.7 W/m^2 |
| avg_vpd | 6.8e-4 kPa | 4.7e-3 kPa |

`total_precip` matches to floating-point error, as expected for a sum. The four
means differ by the documented unequal-month-length effect: April-October has
months of 30 and 31 days, so mean-of-monthly-means is not mean-of-daily. The
magnitudes are small relative to feature scale (radiation ~5000 W/m^2, so 2.9 is
~0.06%).

### Macro R^2, 5 features vs 35 monthly features

| method | 5-feature | 35-feature | change |
|---|---|---|---|
| `ridge` | -1.722 | -5.484 | worse |
| `mlp` | -1.913 | -2.717 | worse |
| `ridge_support_only` | **+0.001** | **-0.161** | **worse** |
| `ridge_refit` | -1.400 | -3.006 | worse |
| `mlp_finetune` | -0.273 | **-0.139** | better |
| `ccpa_no_climate` | -0.560 | -0.303 | better |
| `ccpa` | -0.413 | **-0.149** | better |

The three answers the spec asked for:

1. **Does any method now beat `ridge_support_only`?** Yes. `mlp_finetune`
   (-0.139) and `ccpa` (-0.149) both pass it (-0.161). On 5 features nothing did.
2. **Does `ccpa` now beat `mlp_finetune`?** Not decisively. Macro means are
   -0.149 vs -0.139 — a 0.010 gap against seed standard deviations of 0.079 and
   0.140. `ccpa` wins 4 of 5 seeds, but that is an artifact of aggregation: at
   the (seed, region) level `ccpa` wins only 11 of 30 cells. The 4/5 comes from
   one seed where `mlp_finetune` scored +0.072 and dragged its own mean. Call
   this a tie.
3. **Did `ridge_support_only` improve?** No — it went from +0.001 to -0.161. This
   is the important one. With 32 support rows and 35 features the local-only
   Ridge is overparameterized (p > n) and overfits; alpha=1.0 is not enough
   regularization at that ratio. So the gap did not close purely because transfer
   got better. `mlp_finetune` improved by 0.134 while `ridge_support_only`
   degraded by 0.162 — the crossing is roughly half real progress, half a
   weakened baseline. **A reviewer will spot this**, so the honest framing is
   "monthly features help transfer methods and hurt the 32-sample local
   baseline," not "transfer now wins."

### Per-region structure (35 features)

`ccpa` is best on Northern Crescent; `mlp_finetune` is best on Heartland,
Northern Great Plains and Prairie Gateway; `ridge_support_only` is best on
Basin & Range; `ridge_refit` on Southern Seaboard. Excluding the Basin & Range
fold (the one that dominates macro averages through extreme negatives — `ridge`
hits -31.7 there), the five-region macro is `mlp_finetune` +0.106, `ccpa`
+0.058, `ccpa_no_climate` -0.072, `ridge_support_only` -0.173. Positive R^2 on
five of six regions is a genuine improvement over the 5-feature run, where
almost everything was negative.

`ccpa` beats its ablation `ccpa_no_climate` on both feature sets (-0.149 vs
-0.303 monthly; -0.413 vs -0.560 seasonal), so FiLM climate conditioning does
contribute. That is the one CCPA-specific claim the data supports.

### Capacity note (flagged, not changed)

At 35 inputs the hidden widths (64, 32) are plausibly a bottleneck, and they
were left untouched as instructed. CCPA is 25,633 parameters and the MLP 4,417;
the first layer compresses 35 -> 64 immediately, so per-month structure has to
survive a near-2x squeeze in one step. Widening is the obvious next ablation,
but it would break comparability with everything recorded so far, so it should
be run as a separate arm rather than an edit to these numbers.

---

## Experiment 2 — support-size sweep

`training/support_sweep.py` sweeps support in {8, 16, 32, 64} across 5 seeds and
6 folds. Within each (seed, fold) the **query set is fixed** to the rows
excluded at the largest support size (64), so all four conditions score on
identical rows; support sets are nested prefixes of one permutation. This is
asserted per fold via query-index checksums. Zero-support `ridge` and `mlp` are
computed once per fold on the same fixed query set and recorded at
`support_size=0` as reference lines.

Because the query set is pinned at s=64, `n_query` differs from the main LOCO
run: Basin & Range drops to 35 rows, and its facet is labeled noisy.

### 5-feature sweep — macro R^2

| method | s=8 | s=16 | s=32 | s=64 |
|---|---|---|---|---|
| `ridge_support_only` | -0.452 | -0.160 | -0.004 | **+0.058** |
| `ridge_refit` | -2.083 | -1.975 | -1.779 | -1.484 |
| `mlp_finetune` | **-0.459** | **-0.431** | **-0.363** | -0.341 |
| `ccpa_no_climate` | -0.631 | -0.669 | -0.658 | -0.655 |
| `ccpa` | -0.475 | -0.498 | -0.482 | -0.438 |

Reference lines: `ridge` -2.202, `mlp` -2.495 (flat, no support).

`ccpa` - `mlp_finetune` by support size: -0.016 (s=8), -0.068 (s=16), -0.119
(s=32), -0.097 (s=64); per-seed wins 2/5, 2/5, 1/5, 2/5. The gap is smallest at
the smallest support, which is the direction the meta-learning hypothesis
predicts, but it is a 0.016 difference against a ~0.15 seed spread and `ccpa`
never actually leads. **This does not support a scarce-label advantage claim.**

The genuinely interesting line is `ridge_support_only`: it is the steepest
curve on the plot, going from worst-but-one at s=8 to best overall at s=64. The
adaptive neural methods are nearly flat in support size — they are not
converting extra local labels into accuracy, which is its own finding about the
10-step adaptation budget.

### 35-feature monthly sweep — macro R^2

| method | s=8 | s=16 | s=32 | s=64 |
|---|---|---|---|---|
| `ridge_support_only` | -0.814 | -0.520 | -0.192 | **+0.118** |
| `ridge_refit` | -6.077 | -5.181 | -4.006 | -2.631 |
| `mlp_finetune` | -0.465 | -0.352 | -0.293 | -0.227 |
| `ccpa_no_climate` | -0.488 | -0.447 | -0.434 | -0.408 |
| `ccpa` | **-0.268** | **-0.210** | **-0.223** | -0.190 |

Reference lines: `ridge` -7.249, `mlp` -3.354.

Here `ccpa` leads at every support size, and the margin over `mlp_finetune`
widens monotonically as support shrinks: +0.037, +0.069, +0.142, +0.196 going
from s=64 down to s=8 (per-seed wins 3/5, 4/5, 4/5, 4/5). Taken at face value
this is the scarce-label advantage the meta-learning story predicts.

**The caveat that must go in the paper.** The macro mean is not robust here.
Per (seed, region) cell, `ccpa` wins only 19/30 at s=8 and 10-11/30 at larger
sizes, and the *median* cell delta is negative at s=16/32/64. Excluding Basin &
Range, `ccpa` loses to `mlp_finetune` at every support size (-0.005, -0.038,
-0.060, -0.063). The entire macro advantage comes from Basin & Range, where the
delta is +1.202 at s=8 — one region with 35 query rows. Region-by-region, `ccpa`
wins Basin & Range and Northern Crescent and loses Heartland, Northern Great
Plains, Prairie Gateway.

The two runs also disagree at s=32 because they score different query sets (the
main run keeps rows after index 32, the sweep pins the query to rows after index
64): main gives `ccpa` - `mlp_finetune` = -0.011, sweep gives +0.069. Both are
internally valid; the disagreement itself shows the aggregate difference is
inside the noise floor. The *trend across support sizes* is the trustworthy part
of the sweep, since all four points share one query set by construction.

Honest summary for (b): CCPA's advantage widens as labels shrink **on the
smallest and most climatically distinct region**, and that is a publishable,
specific claim. It is not a general property across regions.

### Figures

- `results/figures/support_sweep.{pdf,png}` — single-column macro R^2 vs support
  size, +-1 std bands, dashed/dotted zero-support references, labeled R^2 = 0 line.
- `results/figures/support_sweep_by_region.{pdf,png}` — 2x3 per-region facets.
- `_monthly` variants of both for the 35-feature sweep.

On monthly features `ridge_refit` and the zero-support references run to -7,
which would compress every informative series into a sliver, so the y-range is
set from the informative methods and off-scale series are named in a footnote
inside the axes rather than silently cropped.

Colors are the validated categorical palette in fixed slot order (validator:
all checks pass, light mode); the three hues below 3:1 contrast carry direct
end-of-line labels per the relief rule, with marker shape as secondary encoding.

---

## Files

- `results/monthly/loco_raw.csv`, `results/monthly/loco_summary.csv`
- `results/sweep/sweep_raw.csv`, `results/sweep/sweep_summary.csv`
- `results/sweep_monthly/sweep_raw.csv`, `results/sweep_monthly/sweep_summary.csv`
- `results/figures/support_sweep{,_by_region}{.pdf,.png}` and `_monthly` variants
- `results/loco_raw.csv`, `results/loco_summary.csv` (5-feature main run)

Reproduce:

    python data/build_master_monthly.py
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset_monthly.csv --out results/monthly/
    python training/support_sweep.py --seeds 0 1 2 3 4 --data master_dataset.csv --out results/sweep/
    python training/support_sweep.py --seeds 0 1 2 3 4 --data master_dataset_monthly.csv --out results/sweep_monthly/
    python training/make_sweep_figure.py --raw results/sweep/sweep_raw.csv --out results/figures/
