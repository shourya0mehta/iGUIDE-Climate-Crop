"""Ceiling-normalized transfer regression (results-freeze Item 1).

The raw ceiling<->transfer correlation is partly mechanical: both R^2s share
the held-out region's yield-variance denominator, so a noisy region depresses
both for the same reason. This analysis removes that channel by regressing the
TRANSFER EFFICIENCY RATIO

    efficiency = transfer_R2 / ceiling_R2        (ceiling_R2 >= 0.05 required)

on the two distance measures only (energy distance; label Wasserstein; MMD as
a robustness check). Ceiling moves to the denominator and is no longer a
predictor. Cells with ceiling < 0.05 are EXCLUDED, not clipped (the ratio is
meaningless there); exclusions are reported explicitly. Efficiency may be
negative (transfer actively harmful where local prediction works) and is not
clipped.

Inference on n = 22 seed-averaged fold cells (primary; per-seed cells with a
cluster bootstrap by region as a check): bootstrap percentile CIs (10k, seed
0) plus exact two-sided PERMUTATION p-values (10k shuffles of the dependent
variable, seed 0, p = (1 + #{|r_perm| >= |r_obs|}) / (N+1)). Because the
efficiency distribution is heavy-tailed, leave-one-out influence is reported:
the max |delta r| over single-cell deletions, and the named most influential
cell. The permutation test for the ceiling<->transfer headline of Section 3
is also computed here so FINDINGS can quote it inline.

Usage:
    python training/efficiency_regression.py --out results/shift/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.partition_analysis import (  # noqa: E402
    FEATURE_SETS, RESPONSE_METHOD, load_runs, standardized_ols)

CEILING_MIN = 0.05
N_BOOT = 10000
N_PERM = 10000
EFF_PREDICTORS = ["energy_dist", "wass_yield"]
ROBUSTNESS_PREDICTOR = "mmd_rbf"


def perm_p(x, y, stat, rng, reps=N_PERM):
    obs = stat(x, y)
    hits = 0
    for _ in range(reps):
        if abs(stat(x, rng.permutation(y))) >= abs(obs):
            hits += 1
    return obs, (1 + hits) / (reps + 1)


def boot_ci(fn, n, reps=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        v = fn(rng.integers(0, n, n))
        if v is not None and np.isfinite(v):
            vals.append(v)
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def loo_influence(x, y):
    """Max |delta Pearson r| over single-cell deletions; returns (dr, idx)."""
    base = pearsonr(x, y)[0]
    drops = [abs(pearsonr(np.delete(x, i), np.delete(y, i))[0] - base)
             for i in range(len(x))]
    i = int(np.argmax(drops))
    return drops[i], i


def univariate_report(cells, pred, yname="efficiency"):
    x, y = cells[pred].values, cells[yname].values
    r_p, p_p = perm_p(x, y, lambda a, b: pearsonr(a, b)[0],
                      np.random.default_rng(0))
    r_s, p_s = perm_p(x, y, lambda a, b: spearmanr(a, b)[0],
                      np.random.default_rng(0))
    lo, hi = boot_ci(lambda idx: pearsonr(x[idx], y[idx])[0]
                     if len(np.unique(x[idx])) > 1 else None, len(x))
    dr, i = loo_influence(x, y)
    who = "%s/%s" % (cells.iloc[i]["partition"], cells.iloc[i]["held_out_region"])
    x2, y2 = np.delete(x, i), np.delete(y, i)
    print("  %-11s pearson %+0.3f [%+0.3f, %+0.3f] perm p=%.4f | "
          "spearman %+0.3f perm p=%.4f" % (pred, r_p, lo, hi, p_p, r_s, p_s))
    print("               max single-cell influence: %s (|dr|=%.3f; "
          "without it pearson %+0.3f, spearman %+0.3f)"
          % (who, dr, pearsonr(x2, y2)[0], spearmanr(x2, y2)[0]))


def build_cells(raw, shift, features):
    resp = raw[(raw["method"] == RESPONSE_METHOD) & (raw["features"] == features)]
    per_seed = resp.merge(shift, on=["partition", "held_out_region"],
                          validate="many_to_one")
    mean_cells = (per_seed.groupby(["partition", "held_out_region"])
                  .agg(r2=("r2", "mean"), r2_std=("r2", "std"),
                       energy_dist=("energy_dist", "first"),
                       mmd_rbf=("mmd_rbf", "first"),
                       wass_yield=("wass_yield", "first"),
                       inregion_r2=("inregion_r2", "first")).reset_index())
    return per_seed, mean_cells


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/shift/")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)

    raw = load_runs()
    shift = pd.read_csv(out_dir / "shift_measures.csv")

    for features in FEATURE_SETS[::-1]:  # monthly (primary) first
        per_seed, cells = build_cells(raw, shift, features)
        print("\n%s\n%s FEATURES (%s)\n%s" % ("=" * 72, features.upper(),
              "primary" if features == "monthly" else "robustness", "=" * 72))

        excluded = cells[cells["inregion_r2"] < CEILING_MIN]
        print("exclusion rule ceiling < %.2f: %d of %d cells excluded"
              % (CEILING_MIN, len(excluded), len(cells)))
        for _, row in excluded.iterrows():
            print("  EXCLUDED %s/%s (ceiling %+0.3f)"
                  % (row["partition"], row["held_out_region"], row["inregion_r2"]))
        kept = cells[cells["inregion_r2"] >= CEILING_MIN].reset_index(drop=True)
        kept["efficiency"] = kept["r2"] / kept["inregion_r2"]
        print("smallest retained ceiling: %.3f (%s/%s)"
              % (kept["inregion_r2"].min(),
                 kept.loc[kept["inregion_r2"].idxmin(), "partition"],
                 kept.loc[kept["inregion_r2"].idxmin(), "held_out_region"]))

        if features == "monthly":
            kept.to_csv(out_dir / "efficiency_cells.csv", index=False)
            print("wrote %s" % (out_dir / "efficiency_cells.csv"))

        print("\nefficiency distribution: min %+0.2f  median %+0.2f  max %+0.2f"
              % (kept["efficiency"].min(), kept["efficiency"].median(),
                 kept["efficiency"].max()))

        print("\nunivariate vs efficiency (n=%d seed-averaged cells):" % len(kept))
        for pred in EFF_PREDICTORS + [ROBUSTNESS_PREDICTOR]:
            univariate_report(kept, pred)

        X = kept[EFF_PREDICTORS].values
        y = kept["efficiency"].values
        betas = standardized_ols(X, y)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            b = standardized_ols(X[idx], y[idx])
            if b is not None:
                boot.append(b)
        boot = np.asarray(boot)
        print("\nmultiple regression (energy + wasserstein), standardized betas:")
        for j, pred in enumerate(EFF_PREDICTORS):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            print("  %-11s beta %+0.3f [%+0.3f, %+0.3f]" % (pred, betas[j], lo, hi))

        # per-seed cells with cluster bootstrap by region (secondary)
        ps = per_seed[per_seed["inregion_r2"] >= CEILING_MIN].copy()
        ps["efficiency"] = ps["r2"] / ps["inregion_r2"]
        clusters = (ps["partition"] + "/" + ps["held_out_region"]).values
        uniq = pd.unique(clusters)
        members = {c: np.flatnonzero(clusters == c) for c in uniq}
        print("\nper-seed cells (n=%d), cluster bootstrap by region:" % len(ps))
        for pred in EFF_PREDICTORS:
            x_all, y_all = ps[pred].values, ps["efficiency"].values
            r = pearsonr(x_all, y_all)[0]

            def stat(unit_idx):
                rows = np.concatenate([members[uniq[i]] for i in unit_idx])
                if len(np.unique(x_all[rows])) < 2:
                    return None
                return pearsonr(x_all[rows], y_all[rows])[0]
            lo, hi = boot_ci(stat, len(uniq))
            print("  %-11s pearson %+0.3f [%+0.3f, %+0.3f]" % (pred, r, lo, hi))

        # Section-3 headline permutation test: ceiling vs raw transfer R2
        x, y = cells["inregion_r2"].values, cells["r2"].values
        r_p, p_p = perm_p(x, y, lambda a, b: pearsonr(a, b)[0],
                          np.random.default_rng(0))
        r_s, p_s = perm_p(x, y, lambda a, b: spearmanr(a, b)[0],
                          np.random.default_rng(0))
        print("\n[Section-3 headline, all %d cells] ceiling vs raw transfer R^2: "
              "pearson %+0.3f perm p=%.4f | spearman %+0.3f perm p=%.4f"
              % (len(cells), r_p, p_p, r_s, p_s))
        x, y = cells["energy_dist"].values, cells["r2"].values
        _, p_e = perm_p(x, y, lambda a, b: pearsonr(a, b)[0],
                        np.random.default_rng(0))
        x, y = cells["wass_yield"].values, cells["r2"].values
        _, p_w = perm_p(x, y, lambda a, b: pearsonr(a, b)[0],
                        np.random.default_rng(0))
        print("[raw-R^2 permutation p, for completeness] energy %.4f | "
              "wasserstein %.4f" % (p_e, p_w))


if __name__ == "__main__":
    main()
