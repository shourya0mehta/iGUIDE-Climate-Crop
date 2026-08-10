# FINDINGS — partition sensitivity and shift decomposition

*2026-08-10. All results: corrected LOCO harness, 5 seeds, all leakage guards
active and passing, CPU-only, byte-identical reruns. This file supersedes parts
of `results/EXPERIMENTS.md` (banner added there). Reproduction commands at the
bottom.*

**Thesis being tested.** Conclusions in crop-yield OOD benchmarks depend on the
choice of spatial partition. The USDA Farm Resource Regions used in this
literature are administrative/economic units, but the shift being studied is
climatic. We run the identical harness under three partitions of the same
county-years and then decompose *why* transfer fails.

---

## 1. What changed since the last results

**(a) The ridge baselines were under-regularized, and fixing them kills the
previous headline claim.** `ridge_support_only` and `ridge_refit` used
`Ridge(alpha=1.0)`; with 35 features and 8–32 support rows that is p > n with a
near-zero penalty. Both now select alpha by internal efficient-LOO CV
(`RidgeCV`, grid `logspace(-2, 4, 25)`), on fitting rows only, never on query.
The CV picks alpha ≈ 32 (median; up to 1778) for the local-only model at s=32 —
the fixed penalty was ~30× too weak.

| monthly features, state partition | fixed α=1.0 (old) | RidgeCV (new) |
|---|---|---|
| `ridge_support_only`, LOCO s=32 macro R² | −0.161 | **+0.009** |
| `ridge_support_only`, sweep s=8 | −0.814 | **−0.242** |
| `ridge_support_only`, sweep s=16 | −0.520 | **−0.153** |
| `ridge_support_only`, sweep s=64 | +0.118 | **+0.202** |
| `ridge_refit`, LOCO s=32 macro R² | −3.006 | −3.005 (α≈1.8; pool swamps 32 rows) |

**Superseded:** EXPERIMENTS.md's strongest claim — *"transfer wins when local
labels are scarce, crossover around 32–64"* — is dead. On monthly features the
properly regularized local-only baseline now beats or ties every transfer
method at every support size (at s=8 it is −0.242 ± 0.266 vs `ccpa` −0.268 ±
0.216: a tie). The only surviving remnant is on 5 seasonal features at s=8
(`mlp_finetune` −0.459 vs local-only −0.633 ± 0.609), inside noise. Old
fixed-α outputs are retained at `results/monthly/`, `results/sweep*/` as a
robustness reference; RidgeCV-era results live in `results/partitions/`,
`results/sweep_cv/`, `results/sweep_monthly_cv/`. Non-ridge numbers were
unaffected (verified: the splice reproduces every stored ridge cell to
rtol 1e-9 before recomputing; torch methods never touch the changed code path).

**(b) The region definitions were wrong, and not slightly.** Regions had been
assigned by state; real ERS Farm Resource Regions are county-level and cross
state lines. Using the authoritative USDA/ERS crosswalk (`data/reglink.xls`,
"Aggregating counties to ERS resource regions", downloaded from ers.usda.gov,
sha256 in `data/build_partitions.py`; 3,112 counties, exactly 9 regions; spot
checks Champaign IL → Heartland, Bolivar MS → Mississippi Portal, Wise VA →
Eastern Uplands all pass; all 8,691 of our county-years match):

- **44.5% of county-years change region** under the correct map.
- The old **"Basin and Range" fold — the single region that carried CCPA's
  entire scarce-label advantage in the old sweep — was 30/31 counties actually
  Fruitful Rim.** Under the corrected map that story is moot: on the true
  Fruitful Rim fold `ccpa` and `mlp_finetune` tie (0.332 vs 0.325).
- Three of the nine ERS regions (Eastern Uplands, Mississippi Portal, Fruitful
  Rim) were entirely missing from the old partition.
- Inclusion rule (≥150 county-years per fold): true Basin and Range has only
  **9** corn county-years (4 counties) and is dropped, loudly. 8 regions
  survive; the ERS datasets have 8,682 rows (−9).

**(c) Three partitions, identical harness.** Same methods, seeds,
hyperparameters, and assertions throughout; only the fold variable changes:
`region` (state-level approximation, 6 folds), `region_ers` (county-level ERS,
8 folds), `region_clim` (k-means k=8 — matched to surviving ERS regions — on
standardized county normals of the 35 monthly features, `n_init=10`,
`random_state=0`, inertia 25,752; X only, never yield; every county keeps one
cluster across years, so it is a spatial partition). Cluster sizes 320–2,958
county-years — all clear the 150 threshold. Identities: Clim-0 transitional
corn belt (KY/IL/MO/IN), Clim-1 core northern belt (IA/MN/MI/OH), Clim-2 humid
Southeast, Clim-3 cool/wet Northeast, Clim-4 dry plains (NE/KS/SD/ND), Clim-5
hot/wet coastal South, Clim-6 arid high-VPD West (KS/TX/CO/ID; humidity z=−3.2),
Clim-7 hot southern plains (TX/OK/KS/AR; mean yield 93 bu/acre vs ~157 overall).

---

## 2. Partition sensitivity

Figure: `results/figures/partition_method_heatmap.{pdf,png}`.

Macro R² (equal-weight over held-out folds, mean over 5 seeds):

| method | state 5f | ers 5f | clim 5f | state 35f | ers 35f | clim 35f |
|---|---|---|---|---|---|---|
| `ridge` | −1.72 | −0.77 | −0.55 | −5.48 | −0.81 | −0.43 |
| `mlp` | −1.91 | −0.73 | −0.22 | −2.72 | −0.32 | **+0.02** |
| `ridge_support_only` | +0.04 | +0.09 | −0.04 | +0.01 | +0.11 | −0.11 |
| `ridge_refit` | −1.41 | −0.74 | −0.50 | −3.00 | −0.66 | −0.31 |
| `mlp_finetune` | −0.27 | −0.10 | **−0.02** | −0.14 | +0.05 | **+0.19** |
| `ccpa_no_climate` | −0.56 | −0.48 | −0.25 | −0.30 | +0.01 | +0.10 |
| `ccpa` | −0.41 | −0.30 | −0.11 | −0.15 | **+0.17** | +0.17 |

**2.1 The method ranking flips with the partition.** On monthly features the
top method is `ridge_support_only` under the state partition (+0.01), **`ccpa`
under county ERS (+0.17)**, and `mlp_finetune` under climate clusters (+0.19).
Spearman rank correlation of method orderings between partitions: 0.89
(state↔ers), 0.64 (state↔clim), 0.71 (ers↔clim) on monthly; 0.75–0.93 on
seasonal. Even "does anything beat the local-only baseline?" flips: no under
state, yes under ers (`ccpa` +0.165 > +0.113) and clim (`mlp_finetune` +0.192,
`ccpa` +0.171, even zero-shot `mlp` +0.021 > −0.113).

**2.2 The hardest fold changes identity.** By `mlp_finetune` mean R²: state →
Basin and Range (−1.36 monthly); ers → **Mississippi Portal** (−0.27); clim →
Clim-7, hot southern plains (−0.62). These are not the same kind of place: the
state partition's hardest fold is the (mislabeled) arid West; the corrected
map's hardest fold is the Delta — which the state partition *cannot see*,
because 591 of the state-level "Southern Seaboard"'s 2,681 county-years are
actually Mississippi Portal and their difficulty is diluted into a bigger,
easier fold.
The one commonality across partitions: every partition's worst *zero-shot*
fold is arid/irrigation-dominated (ridge: B&R −31.7; Fruitful Rim −2.96;
Clim-6 −2.49).

**2.3 The apparent size of the OOD problem is partition-dependent.** Zero-shot
`ridge` macro R² on identical data: −5.48 (state) → −0.81 (ers) → −0.43
(clim); `mlp` −2.72 → −0.32 → +0.02. The state-level benchmark **overstates**
OOD difficulty, largely via the mislabeled Basin & Range fold. Climate-
clustered folds are the easiest for pool-trained methods — but *harder* for
the local-only baseline (+0.11 ers → −0.11 clim), because homogeneous-climate
folds reward pooled training more than 32 local labels. Caveat: macro averages
over different fold structures (6 vs 8 folds, different query sets), so
cross-partition magnitudes describe the benchmark, not method quality.

**2.4 The meta-learning conclusion is itself partition-dependent.**
`ccpa` − `mlp_finetune` macro gap (monthly): state −0.011 (4/5 seed wins, a
tie), **ers +0.115 (5/5 seed wins, and 8/8 regions)**, clim −0.021 (1/5). On
seasonal features ccpa loses under all three (0–1/5 wins). Two things follow:
(i) correcting the region definitions moved the comparison *in favor* of the
novel method — the opposite of the usual suspicion; (ii) no partition-free
claim like "meta-learning fixes geographic transfer" is supportable — the
defensible statement must name its partition. `ccpa` > `ccpa_no_climate` in
all six runs (gaps +0.069 to +0.174; largest on ers seasonal), so FiLM climate
conditioning robustly contributes.

---

## 3. Shift decomposition

Figures: `results/figures/shift_vs_transfer.{pdf,png}`,
`results/figures/coef_cosine_heatmap.{pdf,png}`. Per-fold measures:
`results/shift/shift_measures.csv` (22 folds; monthly features throughout).

Measures per (partition, held-out fold): **energy distance** and **RBF-MMD**
(median-heuristic) between pool and fold features, standardized by the pool
scaler (the two agree, r = 0.945 — MMD adds nothing and is not discussed
further); **1-D Wasserstein** between pool and fold yield distributions
(bu/acre); **in-region ceiling** = 5-fold CV R² of RidgeCV within the fold
only; **coefficient cosine** between per-region RidgeCV fits on globally
standardized features.

**3.1 A structural premise did not survive measurement.** The design assumed
climate-clustered folds minimize covariate shift. They do not: mean energy
distance is clim 4.32 vs ers 2.44 vs state 4.76. Holding out a compact climate
cluster removes a whole neighborhood of feature space, so the fold's *distance
to the pool* stays large even though the fold is internally homogeneous. What
the three partitions actually provide is *variation* in covariate distance
(0.48–16.2 across the 22 folds) — enough spread to test which shift measure
predicts failure, which is the analysis that carries the argument below.

**3.2 Which shift predicts transfer failure?** Response: `mlp_finetune` R² per
fold (seed-averaged, monthly runs; n = 22). Bootstrap 95% CIs, 10k resamples:

| predictor | univariate Pearson [CI] | Spearman | multiple-regression β [CI] |
|---|---|---|---|
| energy distance (covariate) | −0.68 [−0.89, +0.01] | −0.22 | −0.38 [−0.62, +0.03] |
| Wasserstein yield (label) | −0.49 [−0.74, +0.14] | −0.28 | −0.20 [−0.48, +0.11] |
| in-region ceiling (concept/intrinsic) | **+0.74 [+0.42, +0.89]** | +0.64 | **+0.51 [+0.30, +0.76]** |

The in-region ceiling is the **only predictor whose CI excludes zero**, in the
univariate and the multiple regression, on both feature sets (seasonal:
r = +0.81, β = +0.75 [+0.55, +0.94]), and under the per-seed cluster bootstrap
(n = 110 cells, resampled by region). The energy-distance correlation is
point-large but outlier-leveraged: its Spearman is only −0.22, and it hinges on
the two extreme-distance arid folds. Clim-6 is the clean counterexample —
second-highest covariate distance (12.9) yet `mlp_finetune` −0.155 and
local-only +0.131. Clipping R² at −2 changes nothing (no response cell is
below it).

**3.3 Reading.** Transfer fails where the *local climate→yield function* is
weak or different, not where the climate is far away, and not primarily where
the yield marginal is shifted:

- **Adjei's label-shift bottleneck (arXiv 2605.08113) is not corroborated
  here.** Label Wasserstein is the weakest predictor on every cut (β −0.20,
  CI spans zero). This is a genuine disagreement with his Africa-LOCO
  interpretation, on a different continent and benchmark — worth reporting as
  such, not smoothing over.
- **Concept shift is directly visible in the coefficients.** Mean off-diagonal
  cosine between per-region coefficient vectors is only 0.15 (state), 0.17
  (ers), 0.28 (clim). Prairie Gateway's coefficients are *anti-correlated*
  with Heartland's (−0.52). Fruitful Rim is the smoking gun: in-region ceiling
  0.79 (the most self-predictable region in the study) with coefficient cosine
  ≈ 0 to every other region and to the pool (−0.01): the same climate features
  predict its yields by a *different function* — consistent with irrigation
  decoupling yield from precipitation. Zero-shot ridge scores −2.96 there
  while 32 local labels score +0.74.
- The ceiling result has an honest double meaning (see §5): a low ceiling can
  be concept difference *or* irreducible local noise; the coefficient-cosine
  evidence is what separates the two for the arid folds.

---

## 4. What is now defensible (one sentence each, with the number)

1. **Benchmark conclusions in this setting are partition-dependent:** the top
   method under the state, county-ERS, and climate partitions is
   `ridge_support_only` (+0.01), `ccpa` (+0.17), and `mlp_finetune` (+0.19)
   respectively (monthly features; ranking Spearman down to 0.64).
2. **The state-level FRR approximation used previously misassigns 44.5% of
   county-years,** and its most influential fold (Basin & Range) was 30/31
   counties mislabeled — every prior conclusion resting on that fold is void.
3. **The scarce-label transfer claim was an artifact of a mis-set ridge
   penalty:** with CV-chosen regularization the local-only baseline improves
   from −0.814 to −0.242 at s=8 (monthly) and beats or ties every transfer
   method at every support size.
4. **Whether meta-learning beats fine-tuning depends on the partition:**
   `ccpa` − `mlp_finetune` = +0.115 (5/5 seeds, 8/8 regions) under county ERS,
   −0.021 (1/5) under climate clusters, −0.011 (tie) under state.
5. **FiLM climate conditioning helps in all six runs:** `ccpa` beats
   `ccpa_no_climate` by +0.07 to +0.17 macro R² everywhere.
6. **Transfer failure is predicted by the held-out region's own
   predictability, not by its climatic distance or yield-marginal distance:**
   in-region ceiling r = +0.74 [+0.42, +0.89], β = +0.51 [+0.30, +0.76] — the
   only CI excluding zero; label Wasserstein is weakest (β −0.20), so the
   label-shift-as-bottleneck hypothesis is not corroborated on this benchmark.
7. **Every partition contains an arid/irrigation-dominated fold where the
   climate→yield mapping inverts** (coefficient cosine ≈ 0 or negative;
   zero-shot ridge −31.7 / −2.96 / −2.49) while local labels work — direct
   evidence that the binding constraint there is concept shift, not covariate
   shift.

## 5. What is weak or unresolved

- **n = 22 folds** in the regression; CIs are wide and honest, but this is a
  correlational analysis over partially overlapping pools (the three
  partitions reuse the same 8,691 county-years), not 22 independent draws.
- **The ceiling–transfer correlation partly reflects shared irreducible
  noise:** a region with noisy yields depresses both its own CV R² and any
  transfer R² measured on it. The coefficient-cosine evidence separates
  "different function" from "noisy region" only for the arid folds; for
  mid-ceiling regions the two are confounded. A variance-adjusted ceiling (or
  year-held-out within-region CV) would tighten this.
- **The climate partition did not do what it was designed to do** (minimize
  fold↔pool covariate shift, §3.1). The regression rescues the argument, but
  a partition explicitly constructed to minimize pool distance (e.g., holding
  out spatially scattered counties matched on climate) is the right next
  experiment and would make the covariate-shift null much cleaner.
- **Cross-partition macro comparisons average different fold structures**
  (6 vs 8 folds, different query sets); directions are robust, magnitudes are
  not strictly commensurable. The ERS datasets also drop 9 Basin-and-Range
  rows (0.1%).
- **The Chakravarty comparison (arXiv 2510.07350) is still not exactly
  head-to-head:** they use seven Farm Resource Regions (which two of the nine
  are dropped is not stated in the abstract), satellite + weather features,
  and different years. What we can now say precisely is *our own* benchmark's
  sensitivity to the partition their setup relies on.
- **RidgeCV at s=8 on 5 features is unstable** (chosen α spans 0.01–10⁴;
  −0.633 ± 0.609): the s=8 seasonal numbers should not be leaned on.
- **Architecture capacity flag remains** (from EXPERIMENTS.md): hidden widths
  (64, 32) were held fixed for comparability at 35 inputs; widening is a
  separate arm.
- Single crop (corn), 2017–2022, climate-only features; none of this speaks
  to satellite-feature pipelines directly.

---

### Files and reproduction

Results: `results/partitions/{state,ers,clim}_{seasonal,monthly}/`,
`results/sweep_cv/`, `results/sweep_monthly_cv/`, `results/shift/`,
`results/figures/`. Fixed-α reference: `results/monthly/`, `results/sweep/`,
`results/sweep_monthly/`, `results/loco_*.csv`.

    python data/build_partitions.py
    python training/splice_ridgecv.py --mode loco  --data master_dataset.csv         --raw results/loco_raw.csv                 --out results/partitions/state_seasonal/
    python training/splice_ridgecv.py --mode loco  --data master_dataset_monthly.csv --raw results/monthly/loco_raw.csv         --out results/partitions/state_monthly/
    python training/splice_ridgecv.py --mode sweep --data master_dataset.csv         --raw results/sweep/sweep_raw.csv          --out results/sweep_cv/
    python training/splice_ridgecv.py --mode sweep --data master_dataset_monthly.csv --raw results/sweep_monthly/sweep_raw.csv  --out results/sweep_monthly_cv/
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset_ers.csv          --region-col region_ers  --out results/partitions/ers_seasonal/
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset_monthly_ers.csv  --region-col region_ers  --out results/partitions/ers_monthly/
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset_clim.csv         --region-col region_clim --out results/partitions/clim_seasonal/
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset_monthly_clim.csv --region-col region_clim --out results/partitions/clim_monthly/
    python training/shift_decomposition.py --out results/shift/
    python training/partition_analysis.py  --out results/shift/
    python training/make_findings_figures.py --out results/figures/
    python training/make_sweep_figure.py --raw results/sweep_monthly_cv/sweep_raw.csv --out results/figures/ --suffix _monthly_cv
