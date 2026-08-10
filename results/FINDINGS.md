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

**Table 1** — macro R² (equal-weight over held-out folds), mean ± std over 5
seeds. Tidy version: `results/main_table.csv` (regenerate with
`python training/make_main_table.py`). Best method per column in bold.

| method | seas. state | seas. ers | seas. clim | mon. state | mon. ers | mon. clim |
|---|---|---|---|---|---|---|
| `ridge` | −1.722 ± .270 | −0.773 ± .006 | −0.553 ± .012 | −5.484 ± 1.078 | −0.810 ± .012 | −0.429 ± .017 |
| `mlp` | −1.913 ± .496 | −0.725 ± .090 | −0.223 ± .055 | −2.717 ± 1.275 | −0.324 ± .074 | +0.021 ± .063 |
| `ridge_support_only` | **+0.043 ± .020** | **+0.089 ± .024** | −0.036 ± .035 | **+0.009 ± .182** | +0.113 ± .043 | −0.113 ± .348 |
| `ridge_refit` | −1.408 ± .223 | −0.740 ± .006 | −0.502 ± .015 | −3.005 ± .653 | −0.657 ± .028 | −0.311 ± .021 |
| `mlp_finetune` | −0.273 ± .115 | −0.104 ± .029 | **−0.019 ± .022** | −0.139 ± .140 | +0.050 ± .028 | **+0.192 ± .029** |
| `ccpa_no_climate` | −0.560 ± .172 | −0.477 ± .020 | −0.251 ± .021 | −0.303 ± .075 | +0.008 ± .059 | +0.102 ± .017 |
| `ccpa` | −0.413 ± .058 | −0.303 ± .042 | −0.111 ± .024 | −0.149 ± .079 | **+0.165 ± .030** | +0.171 ± .028 |

Fold structure the macros average over (n_query at s=32, monthly): state 6
folds (67–2,649 per fold), ers 8 folds (204–2,741), clim 8 folds (288–2,926);
full per-fold sizes are printed by `make_main_table.py`.

**2.1 The method ranking flips with the partition.** On monthly features the
top method is `ridge_support_only` under the state partition (+0.01), **`ccpa`
under county ERS (+0.17)**, and `mlp_finetune` under climate clusters (+0.19).
Spearman rank correlation of method orderings between partitions
(descriptive, computed on the seed-mean macro table): 0.89 (state↔ers), 0.64
(state↔clim), 0.71 (ers↔clim) on monthly; 0.75–0.93 on seasonal. Even "does anything beat the local-only baseline?" flips: no under
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
defensible statement must name its partition. **The 5/5 ERS result is
evidence of conclusion fragility, not a method win:** the same architecture's
apparent value appears and disappears with the choice of map. `ccpa` > `ccpa_no_climate` in
all six runs (gaps +0.069 to +0.174; largest on ers seasonal), so FiLM climate
conditioning robustly contributes.

---

## 3. Shift decomposition

Figures: `results/figures/shift_vs_transfer.{pdf,png}`,
`results/figures/efficiency_scatter.{pdf,png}`,
`results/figures/coef_cosine_heatmap.{pdf,png}`. Per-fold measures:
`results/shift/shift_measures.csv`, `results/shift/efficiency_cells.csv`
(22 folds; monthly features throughout).

Measures per (partition, held-out fold): **energy distance** and **RBF-MMD**
(median-heuristic) between pool and fold features, standardized by the pool
scaler (the two agree, r = 0.95 [0.92, 0.98] across the 22 folds — MMD is
reported as a robustness check only); **1-D Wasserstein** between pool and
fold yield distributions (bu/acre); **in-region ceiling** = 5-fold CV R² of
RidgeCV within the fold only; **coefficient cosine** between per-region
RidgeCV fits on globally standardized features.

**3.1 A caveat before the headline: raw ceiling↔transfer correlation is
partly mechanical.** Transfer R² and in-region R² share the held-out region's
yield-variance denominator, so a region with high unexplainable variance
(management noise, weather-decoupled irrigation) is pushed down on *both*
axes for the same arithmetic reason. The raw correlation in §3.2 therefore
overstates how much "intrinsic predictability" explains; §3.3 removes the
shared-denominator channel by analyzing the ceiling-normalized efficiency
ratio, and the two must be read together.

**3.2 The raw association.** Response: `mlp_finetune` R² per fold
(seed-averaged, monthly runs; n = 22). Bootstrap 95% CIs (10k resamples);
exact two-sided permutation p (10k shuffles, seeded):

| predictor | univariate Pearson [CI] | Spearman | multiple-regression β [CI] |
|---|---|---|---|
| energy distance (covariate) | −0.68 [−0.89, +0.01], perm p = 0.007 | −0.22 | −0.38 [−0.62, +0.03] |
| Wasserstein yield (label) | −0.49 [−0.74, +0.14], perm p = 0.028 | −0.28 | −0.20 [−0.48, +0.11] |
| in-region ceiling (concept/intrinsic) | **+0.74 [+0.42, +0.89], perm p = 0.0001** | +0.64 (perm p = 0.0016) | **+0.51 [+0.30, +0.76]** |

The in-region ceiling is the only predictor whose CI excludes zero, in the
univariate and the multiple regression, on both feature sets (seasonal:
r = +0.81 [+0.59, +0.92], perm p = 0.0001; β = +0.75 [+0.55, +0.94]), and
under the per-seed cluster bootstrap (n = 110 cells, resampled by region).
Clipping R² at −2 changes nothing (no response cell is below it). Subject to
§3.1, this says: *how much* transfer achieves tracks *how much is achievable
locally*.

**3.3 Ceiling-normalized transfer efficiency (does distance explain the
rest?).** For each fold, efficiency = transfer R² / ceiling R², computed only
where ceiling ≥ 0.05 (the ratio is meaningless near zero; negative efficiency
is allowed and means transfer is actively harmful where local prediction
works). **Zero of the 22 cells are excluded by the rule** — the smallest
ceiling in the study is 0.088 (state Basin & Range). Notably, the expectation
that Fruitful Rim would drop out was wrong in an informative way: it has the
*highest* ceiling (0.787) — its transfer problem is a different local
function, not an unpredictable region. Efficiency spans −15.6 (state B&R:
achievable 0.088, delivered −1.36) to +0.90 (Clim-1), median +0.33.

Regressing efficiency on the two distance measures (ceiling now in the
denominator, no longer a predictor):

| predictor | Pearson [CI], perm p | Spearman, perm p | without state/B&R | β [CI] |
|---|---|---|---|---|
| energy distance | −0.76 [−0.97, −0.04], p = 0.004 | −0.26, p = 0.25 | Pearson −0.29, Spearman −0.15 | −0.67 [−1.00, +0.09] |
| Wasserstein yield | −0.48 [−0.84, +0.13], p = 0.053 | −0.35, p = 0.10 | Pearson −0.59, Spearman −0.27 | −0.24 [−0.81, +0.21] |

**Outcome: no leverage-robust distance signal survives.** The energy-distance
Pearson looks decisive (perm p = 0.004) but is manufactured by the single
state/Basin & Range cell — the fold already known to be 30/31 counties
mislabeled — which pairs the study's largest covariate distance (16.2) with
its most extreme efficiency (−15.6, on the smallest retained ceiling). Remove
that one cell and the correlation drops to −0.29; the rank correlation was
never there (−0.26, p = 0.25); both multiple-regression CIs span zero; on
seasonal features the Spearman is −0.04. Label Wasserstein is weaker at the
point estimate but more leverage-stable (−0.48 with B&R, −0.59 without;
Spearman −0.35, p = 0.10) — suggestive, short of significance. So the
two-factor story ("ceiling sets what is achievable, distance explains the
capture rate") is **not** supported: once intrinsic predictability is
accounted for, neither climatic nor label-marginal distance carries robust
signal about transfer success in this data. Distance-based transferability
heuristics would not have worked here — and the one analysis cell that makes
distance look predictive is itself an artifact of the incorrect partition,
which is the paper's thesis in miniature.

**3.4 A structural premise did not survive measurement.** The design assumed
climate-clustered folds minimize covariate shift. They do not: mean energy
distance is clim 4.32 vs ers 2.44 vs state 4.76. Holding out a compact climate
cluster removes a whole neighborhood of feature space, so the fold's *distance
to the pool* stays large even though the fold is internally homogeneous. What
the three partitions actually provide is *variation* in covariate distance
(0.48–16.2 across the 22 folds) — the spread the regressions above rely on.

**3.5 Reading.** Transfer fails where the *local climate→yield function* is
weak or different, not measurably because the climate is far away or the yield
marginal is shifted:

- **We do not corroborate the label-shift bottleneck reading of Adjei (arXiv
  2605.08113) in our setting.** Label Wasserstein was the weakest predictor of
  raw transfer R² on every cut (β −0.20 [−0.48, +0.11]) and short of
  significance against efficiency (perm p = 0.053/0.10). n = 22 folds on one
  crop and country cannot refute his Africa-LOCO interpretation; what it can
  say is that label-marginal distance is not the binding constraint *here*.
- **Concept shift is directly visible in the coefficients.** Mean off-diagonal
  cosine between per-region coefficient vectors is only 0.15 (state), 0.17
  (ers), 0.28 (clim). Prairie Gateway's coefficients are *anti-correlated*
  with Heartland's (−0.52). Fruitful Rim is the sharpest case: in-region
  ceiling 0.79 (the most self-predictable region in the study) while its
  coefficient vector bears **essentially no relationship** to the pool's
  (cosine −0.01, i.e. orthogonal — not inverted): the same climate features
  predict its yields by an unrelated function, consistent with irrigation
  decoupling yield from rainfall. Zero-shot ridge scores −2.96 there while 32
  local labels score +0.74.
- A low ceiling can still be concept difference *or* irreducible local noise;
  the coefficient-cosine evidence is what separates the two for the arid
  folds (see §5).

**Multiple-comparisons note.** Across §3.2–3.3 we report three predictors ×
two dependent variables × two distance measures ≈ a dozen tests. Only the
pre-specified headline chain — ceiling vs transfer (§3.2), then distance vs
efficiency (§3.3) — is treated as confirmatory; everything else (MMD
robustness, seasonal repeats, per-seed variants) is exploratory and is
reported for transparency, not inference.

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
6. **Transfer R² tracks the held-out region's own predictability** (in-region
   ceiling r = +0.74 [+0.42, +0.89], perm p = 0.0001; β = +0.51 [+0.30,
   +0.76]) — the only association that survives bootstrap and permutation,
   subject to the shared-denominator caveat stated in §3.1.
7. **Once the ceiling is divided out, no distance measure robustly predicts
   transfer efficiency:** the energy-distance effect (−0.76 [−0.97, −0.04],
   perm p = 0.004) collapses to −0.29 without the single mislabeled Basin &
   Range cell (Spearman −0.26, p = 0.25); label Wasserstein stays suggestive
   at best (−0.48 [−0.84, +0.13], p = 0.053) — so we do not corroborate the
   label-shift bottleneck of Adjei (2605.08113) in this setting, and
   distance-based transferability heuristics carry no robust signal here.
8. **Every partition contains an arid/irrigation-dominated fold where the
   learned climate→yield mapping bears essentially no relationship to the
   pooled one** (coefficient cosine ≈ 0; zero-shot ridge −31.7 / −2.96 /
   −2.49) while local labels work (`ridge_support_only` +0.13 to +0.74) —
   direct evidence that the binding constraint there is concept shift, not
   covariate shift.

## 5. What is weak or unresolved

- **n = 22 folds** in the regression; CIs are wide and honest, but this is a
  correlational analysis over partially overlapping pools (the three
  partitions reuse the same 8,691 county-years), not 22 independent draws.
- **The efficiency ratio is unstable at small ceilings.** The pre-specified
  exclusion floor (0.05) retains state Basin & Range (ceiling 0.088), whose
  efficiency of −15.6 single-handedly produces the nominally significant
  energy-distance Pearson in §3.3; a floor of 0.10 would exclude exactly that
  cell and the distance signal would vanish outright. We report both readings
  rather than tuning the threshold. Even after normalization, "different
  function" vs "noisy region" is separated only by the coefficient-cosine
  evidence, and only cleanly for the arid folds.
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

**Future work (noted, deliberately out of scope for this freeze):**
year-held-out within-region CV as a variance-adjusted ceiling; a partition of
spatially scattered, climate-matched hold-outs to isolate covariate shift
cleanly; widened hidden layers at 35 inputs (capacity arm); satellite/
foundation-model features (Prithvi/Sentinel); an exact seven-region
replication of Chakravarty's fold structure once their region list is known.

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
