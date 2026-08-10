"""Partition-sensitivity stability analyses + shift-vs-transfer regression.

Consumes the six LOCO runs (3 partitions x 2 feature sets) and the shift
measures, and answers, with numbers:

  Phase 2 — conclusion stability across partitions
    1. Does the method ranking flip? (macro-R^2 ranking per partition,
       pairwise Spearman between partitions, top method per partition)
    2. Which held-out fold is hardest, per partition?
    3. Does the magnitude of OOD degradation change for a fixed method?
    4. Does the meta-learning conclusion (ccpa vs mlp_finetune) survive
       under any partition?

  Phase 3 — which shift measure predicts transfer failure?
    Response: mlp_finetune R^2. Predictors: energy_dist (covariate),
    wass_yield (label marginal), inregion_r2 (concept/intrinsic ceiling).
    Primary unit: the (partition, held-out region) cell, seed-averaged
    (n = 22); the per-seed version (n = 110) is reported with a cluster
    bootstrap by region, since seeds share their region's shift measures.
    Bootstrap percentile CIs throughout (10k resamples, seed 0). R^2 is
    unbounded below, so alongside raw Pearson we report Spearman (rank) and
    a floor-clipped robustness variant.

Usage:
    python training/partition_analysis.py --out results/shift/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.loco_eval import METHOD_ORDER  # noqa: E402

PARTITIONS = ["state", "ers", "clim"]
FEATURE_SETS = ["seasonal", "monthly"]
RESPONSE_METHOD = "mlp_finetune"   # strongest transfer method
PREDICTORS = ["energy_dist", "wass_yield", "inregion_r2"]
N_BOOT = 10000
CLIP_FLOOR = -2.0


def load_runs():
    frames = []
    for p in PARTITIONS:
        for f in FEATURE_SETS:
            path = REPO_ROOT / "results" / "partitions" / ("%s_%s" % (p, f)) / "loco_raw.csv"
            raw = pd.read_csv(path)
            raw["partition"], raw["features"] = p, f
            frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def macro_table(raw):
    """Equal-weight macro R^2 mean/std over seeds, per (partition, features, method)."""
    per_seed = (raw.groupby(["partition", "features", "seed", "method"])["r2"]
                .mean().reset_index())
    return (per_seed.groupby(["partition", "features", "method"])["r2"]
            .agg(["mean", "std"]).reset_index())


def q(vals, lo=2.5, hi=97.5):
    return np.percentile(vals, lo), np.percentile(vals, hi)


def boot_stat(fn, n, rng, reps=N_BOOT):
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        v = fn(idx)
        if v is not None and np.isfinite(v):
            out.append(v)
    return np.asarray(out)


def standardized_ols(X, y):
    """OLS on z-scored predictors/response; returns betas (no intercept term
    reported). Degenerate inputs return None."""
    if len(y) < X.shape[1] + 2 or np.any(X.std(axis=0) == 0) or y.std() == 0:
        return None
    Xz = (X - X.mean(axis=0)) / X.std(axis=0)
    yz = (y - y.mean()) / y.std()
    A = np.column_stack([np.ones(len(yz)), Xz])
    beta, *_ = np.linalg.lstsq(A, yz, rcond=None)
    return beta[1:]


def regression_block(cells, label, cluster_col=None):
    """Univariate correlations + standardized multiple OLS with bootstrap CIs.
    cluster_col: if given, bootstrap resamples clusters (regions), not rows."""
    y = cells["r2"].values
    X = cells[PREDICTORS].values
    rng = np.random.default_rng(0)
    print("\n--- %s (n=%d cells) ---" % (label, len(cells)))

    if cluster_col is None:
        def resample(idx):
            return idx
        n_units = len(cells)
    else:
        clusters = cells[cluster_col].values
        uniq = pd.unique(clusters)
        members = {c: np.flatnonzero(clusters == c) for c in uniq}
        n_units = len(uniq)

        def resample(unit_idx):
            return np.concatenate([members[uniq[i]] for i in unit_idx])

    print("univariate vs %s R^2:" % RESPONSE_METHOD)
    for j, pred in enumerate(PREDICTORS):
        r_p = pearsonr(X[:, j], y)[0]
        r_s = spearmanr(X[:, j], y)[0]
        bp = boot_stat(lambda idx: pearsonr(X[resample(idx), j],
                                            y[resample(idx)])[0]
                       if len(np.unique(X[resample(idx), j])) > 1 else None,
                       n_units, np.random.default_rng(0))
        lo, hi = q(bp)
        print("  %-12s pearson %+0.3f  [%+0.3f, %+0.3f]   spearman %+0.3f"
              % (pred, r_p, lo, hi, r_s))

    betas = standardized_ols(X, y)
    boot_betas = []
    rng = np.random.default_rng(0)
    for _ in range(N_BOOT):
        idx = resample(rng.integers(0, n_units, n_units))
        b = standardized_ols(X[idx], y[idx])
        if b is not None:
            boot_betas.append(b)
    boot_betas = np.asarray(boot_betas)
    print("multiple regression, standardized coefficients:")
    for j, pred in enumerate(PREDICTORS):
        lo, hi = q(boot_betas[:, j])
        print("  %-12s beta %+0.3f  [%+0.3f, %+0.3f]" % (pred, betas[j], lo, hi))
    return betas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/shift/")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)

    raw = load_runs()
    shift = pd.read_csv(out_dir / "shift_measures.csv")
    mac = macro_table(raw)

    # ---------------- Phase 2.1: ranking stability ----------------
    print("=" * 72)
    print("2.1  METHOD RANKING BY PARTITION (macro R^2)")
    for f in FEATURE_SETS:
        sub = mac[mac["features"] == f]
        piv = sub.pivot(index="method", columns="partition", values="mean") \
                 .reindex(METHOD_ORDER)[PARTITIONS]
        print("\n[%s features] macro R^2:" % f)
        print(piv.round(3).to_string())
        print("top method: " + ", ".join(
            "%s=%s" % (p, piv[p].idxmax()) for p in PARTITIONS))
        print("Spearman rank correlation between partitions:")
        for a, b in [("state", "ers"), ("state", "clim"), ("ers", "clim")]:
            rho = spearmanr(piv[a], piv[b])[0]
            print("  %s vs %s: rho=%.3f" % (a, b, rho))

    # ---------------- Phase 2.2: hardest region ----------------
    print("\n" + "=" * 72)
    print("2.2  HARDEST HELD-OUT FOLD (by %s mean R^2)" % RESPONSE_METHOD)
    per_region = (raw[raw["method"] == RESPONSE_METHOD]
                  .groupby(["partition", "features", "held_out_region"])["r2"]
                  .mean().reset_index())
    for f in FEATURE_SETS:
        for p in PARTITIONS:
            sub = per_region.query("features == @f and partition == @p") \
                            .sort_values("r2")
            worst = sub.iloc[0]
            print("  [%s %-8s] worst: %-22s R2=%+.3f | best: %-22s R2=%+.3f"
                  % (f, p, worst["held_out_region"], worst["r2"],
                     sub.iloc[-1]["held_out_region"], sub.iloc[-1]["r2"]))

    # ---------------- Phase 2.3: OOD degradation magnitude ----------------
    print("\n" + "=" * 72)
    print("2.3  MACRO R^2 OF FIXED METHODS ACROSS PARTITIONS")
    for f in FEATURE_SETS:
        sub = mac[mac["features"] == f]
        piv = sub.pivot(index="method", columns="partition", values="mean") \
                 .reindex(["ridge", "mlp", "ridge_support_only", "mlp_finetune",
                           "ccpa"])[PARTITIONS]
        print("\n[%s] partition effect on fixed methods:" % f)
        print(piv.round(3).to_string())

    # ---------------- Phase 2.4: meta-learning conclusion ----------------
    print("\n" + "=" * 72)
    print("2.4  CCPA vs MLP_FINETUNE PER PARTITION")
    per_seed = (raw.groupby(["partition", "features", "seed", "method"])["r2"]
                .mean().reset_index())
    for f in FEATURE_SETS:
        for p in PARTITIONS:
            w = per_seed.query("features == @f and partition == @p") \
                        .pivot(index="seed", columns="method", values="r2")
            d = w["ccpa"] - w["mlp_finetune"]
            print("  [%s %-8s] ccpa-mlp_finetune: mean %+0.3f, per-seed wins %d/%d"
                  % (f, p, d.mean(), int((d > 0).sum()), len(d)))

    # ---------------- Phase 3: shift regression ----------------
    print("\n" + "=" * 72)
    print("3    WHICH SHIFT PREDICTS TRANSFER FAILURE?  (response: %s R^2)"
          % RESPONSE_METHOD)
    resp = raw[raw["method"] == RESPONSE_METHOD]
    for f in FEATURE_SETS:
        cells_mean = (resp[resp["features"] == f]
                      .groupby(["partition", "held_out_region"])["r2"]
                      .mean().reset_index()
                      .merge(shift, on=["partition", "held_out_region"],
                             validate="one_to_one"))
        cells_mean.to_csv(out_dir / ("cells_%s.csv" % f), index=False)
        regression_block(cells_mean, "%s features, seed-averaged cells" % f)

        clipped = cells_mean.copy()
        clipped["r2"] = clipped["r2"].clip(lower=CLIP_FLOOR)
        regression_block(clipped, "%s features, R^2 clipped at %.0f "
                         "(outlier robustness)" % (f, CLIP_FLOOR))

        cells_seed = (resp[resp["features"] == f]
                      .merge(shift, on=["partition", "held_out_region"],
                             validate="many_to_one"))
        cells_seed["cluster"] = cells_seed["partition"] + "/" + \
            cells_seed["held_out_region"]
        regression_block(cells_seed, "%s features, per-seed cells, cluster "
                         "bootstrap by region" % f, cluster_col="cluster")

    # covariate-vs-label horse race summary (Adjei hypothesis)
    print("\nNote: energy_dist and mmd_rbf agreement (Pearson across 22 folds): "
          "%.3f" % pearsonr(shift["energy_dist"], shift["mmd_rbf"])[0])


if __name__ == "__main__":
    main()
