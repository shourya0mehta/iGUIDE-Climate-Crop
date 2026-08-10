# iGUIDE Climate-Crop

**Benchmark conclusions in crop-yield OOD prediction are partition-dependent —
and transfer fails for concept-shift reasons, not distance.**

Shourya Mehta | CS + GIS, University of Illinois Urbana-Champaign | svm5@illinois.edu
In collaboration with Dr. Anand Padmanabhan, NSF I-GUIDE Institute

---

## What this repository shows

Out-of-distribution evaluation for county-level crop-yield prediction is
usually organized around USDA Farm Resource Regions — administrative/economic
units — while the shift being studied is climatic. We run one
leakage-guarded Leave-One-Cluster-Out (LOCO) harness over the identical 8,691
corn county-years (2017–2022) under **three definitions of the spatial fold
variable** — the state-level FRR approximation common in practice, the true
county-level ERS regions, and k-means climate clusters — and show that the
benchmark's conclusions (which method wins, which region is hardest, how hard
OOD transfer looks) change with the choice of map. A shift decomposition then
asks *why* transfer fails: the answer implicates the held-out region's own
predictability and region-specific climate→yield functions (concept shift),
not covariate or label-marginal distance.

The full expert-level write-up is **[results/FINDINGS.md](results/FINDINGS.md)**.

## Table 1 — macro R² by method × partition (mean ± std, 5 seeds)

| method | seas. state | seas. ers | seas. clim | mon. state | mon. ers | mon. clim |
|---|---|---|---|---|---|---|
| `ridge` | −1.722 ± .270 | −0.773 ± .006 | −0.553 ± .012 | −5.484 ± 1.078 | −0.810 ± .012 | −0.429 ± .017 |
| `mlp` | −1.913 ± .496 | −0.725 ± .090 | −0.223 ± .055 | −2.717 ± 1.275 | −0.324 ± .074 | +0.021 ± .063 |
| `ridge_support_only` | **+0.043 ± .020** | **+0.089 ± .024** | −0.036 ± .035 | **+0.009 ± .182** | +0.113 ± .043 | −0.113 ± .348 |
| `ridge_refit` | −1.408 ± .223 | −0.740 ± .006 | −0.502 ± .015 | −3.005 ± .653 | −0.657 ± .028 | −0.311 ± .021 |
| `mlp_finetune` | −0.273 ± .115 | −0.104 ± .029 | **−0.019 ± .022** | −0.139 ± .140 | +0.050 ± .028 | **+0.192 ± .029** |
| `ccpa_no_climate` | −0.560 ± .172 | −0.477 ± .020 | −0.251 ± .021 | −0.303 ± .075 | +0.008 ± .059 | +0.102 ± .017 |
| `ccpa` | −0.413 ± .058 | −0.303 ± .042 | −0.111 ± .024 | −0.149 ± .079 | **+0.165 ± .030** | +0.171 ± .028 |

Bold = best method in that column. `seas.` = 5 season-long climate features,
`mon.` = 35 monthly features; partitions are the state-level FRR
approximation (6 folds), county-level ERS regions (8 folds), and climate
clusters (8 folds).

## Headline findings

1. **The winning method depends on the partition** (monthly features):
   local-only ridge under state-level regions (+0.01), CCPA under county ERS
   (+0.17, 5/5 seeds vs fine-tuning), plain fine-tuning under climate
   clusters (+0.19) — rank correlation between partitions as low as 0.64.
   The CCPA result is evidence of conclusion fragility, not a method win.
2. **The commonly used state-level FRR approximation misassigns 44.5% of
   county-years**, and its most influential fold ("Basin and Range") was
   30/31 counties mislabeled; apparent OOD difficulty drops from −5.5 to
   −0.8 macro R² (zero-shot ridge) when the correct county map is used.
3. **A previously reported scarce-label transfer advantage was an artifact of
   a mis-set ridge penalty**: with CV-chosen regularization the local-only
   baseline improves from −0.81 to −0.24 at 8 labels and beats or ties every
   transfer method at every support size.
4. **Transfer fails where the local climate→yield function is weak or
   different, not where the climate is far away**: in-region ceiling
   correlates +0.74 [+0.42, +0.89] (perm p = 0.0001) with transfer R², and
   after normalizing by the ceiling no distance measure survives a
   leverage check; per-region coefficient vectors have mean pairwise cosine
   0.15–0.28, with the irrigated Fruitful Rim orthogonal (−0.01) to the rest.

## Reproduce Table 1 (one command)

```bash
git clone https://github.com/shourya0mehta/iGUIDE-Climate-Crop
cd iGUIDE-Climate-Crop
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python training/make_main_table.py
```

This rebuilds Table 1 from the committed per-seed results in
`results/partitions/` (no model training; the command itself finishes in
about a second — the `pip install` dominates at well under a minute).
`requirements.txt` covers analysis only and works on Python 3.9+.

Re-running the experiments themselves needs `pip install -r
requirements-train.txt` **in a Python 3.9 environment** (learn2learn 0.2.0
does not build on ≥3.11) plus the master datasets (built by
`data/build_master*.py` + `data/build_partitions.py` from USDA/HRRR source
CSVs; the ERS county crosswalk `data/reglink.xls` is vendored). The full
3-partition × 2-feature-set × 5-seed sweep is CPU-only and takes ~40 minutes
wall on an 18-core machine with the four `loco_eval.py` runs in parallel
(a few hours serial); exact commands are at the bottom of
`results/FINDINGS.md`. Runs are seeded and byte-reproducible.

## Repository structure

    data/               dataset + partition builders (USDA, WRF-HRRR, ERS crosswalk)
    models/             MLP baselines and the CCPA architecture (FiLM climate conditioning)
    training/           LOCO harness, support sweep, shift decomposition, analyses, figures
    results/            committed per-seed results, FINDINGS.md, figures
        FINDINGS.md     the write-up (read this)
        EXPERIMENTS.md  earlier experiment log (partly superseded; banner inside)

## Limitations

n = 22 held-out folds across three overlapping partitions of the same data —
correlational, with wide bootstrapped CIs reported inline; a single crop
(corn), a single country (US), 2017–2022; tabular climate features only (no
satellite/foundation-model inputs); the shift-decomposition ceilings use
within-region CV, which shares a variance denominator with transfer R²
(addressed via the efficiency ratio in FINDINGS §3.3, with its own caveats).

## Superseded material

Earlier versions of this repository framed CCPA (a climate-conditioned
meta-learned adapter) as an architectural fix for geographic transfer, on a
state-level region partition with fixed-α ridge baselines. Those claims did
not survive the corrections above and are retained only as history:
`results/EXPERIMENTS.md` carries a superseded banner, fixed-α outputs remain
under `results/monthly/` and `results/sweep*/` as robustness references, and
the foundation-model roadmap (Prithvi/Sentinel-2) is deferred future work.
CCPA itself remains in the comparison as one of seven methods; its FiLM
conditioning beats its ablation in all six runs (+0.07 to +0.17 macro R²).

## References

- USDA ERS, *Farm Resource Regions* (AIB-760, 2000) — county crosswalk
  vendored at `data/reglink.xls`
- Chakravarty (TMLR 2026, arXiv 2510.07350) — LOCO across seven Farm
  Resource Regions with satellite + weather features
- Adjei et al. (2025, arXiv 2605.08113) — Africa LOCO benchmark; label-shift
  bottleneck reading not corroborated here (FINDINGS §3.5)
- Lin et al., *CropNet* (KDD 2024) — source of the USDA yield + WRF-HRRR
  weather county-year data
