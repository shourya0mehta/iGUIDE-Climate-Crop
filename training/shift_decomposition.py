"""Shift decomposition per (partition, held-out region).

For each fold of each spatial partition, quantify the three kinds of shift a
transfer method faces, using the 35 monthly climate features throughout:

Covariate shift
    energy_dist : multivariate energy distance (U-statistic, exact — no
        subsampling) between the training pool and the held-out region,
        features standardized by the TRAIN-POOL scaler exactly as the
        harness does.
    mmd_rbf : RBF-kernel MMD (unbiased U-statistic), median-heuristic
        bandwidth. Sides larger than MMD_MAX_N rows are subsampled with a
        fixed seed (deterministic; noted in the output).

Label marginal shift
    wass_yield : 1-D Wasserstein distance between training-pool and held-out
        yield distributions, in bu/acre.

Concept shift proxies
    inregion_r2 : 5-fold CV R^2 of RidgeCV fit and evaluated WITHIN the
        held-out region only (scaler fit inside each CV fold — no leakage).
        Low ceiling = the region is intrinsically hard, not merely far away.
    coef_cos_pool : cosine similarity between the held-out region's own
        RidgeCV coefficients and the training pool's, both fit on globally
        standardized features so the axes match.

Also writes, per partition, the full pairwise cosine-similarity matrix of
per-region RidgeCV coefficient vectors (globally standardized features):
low similarity = the climate->yield mapping is literally a different
function there — concept shift, not covariate shift.

These are descriptive diagnostics of the benchmark, not part of the
leakage-guarded evaluation; the global scaler used for coefficient
comparison is deliberate and disclosed.

Usage:
    python training/shift_decomposition.py --out results/shift/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.baselines import TARGET_COL  # noqa: E402
from training.loco_eval import RIDGE_CV_ALPHAS, load_data  # noqa: E402

PARTITIONS = [  # (name, monthly dataset, region column)
    ("state", "master_dataset_monthly.csv", "region"),
    ("ers", "master_dataset_monthly_ers.csv", "region_ers"),
    ("clim", "master_dataset_monthly_clim.csv", "region_clim"),
]
MMD_MAX_N = 3000
MEDIAN_HEURISTIC_N = 2000
CV_FOLDS = 5
SEED = 0


def mean_pairwise_dist(A, B=None, chunk=1000):
    """Mean Euclidean distance over all (a, b) pairs; if B is None, the
    U-statistic mean over distinct pairs of A (diagonal excluded). Exact,
    chunked to bound memory."""
    self_mode = B is None
    if self_mode:
        B = A
    total = 0.0
    for i in range(0, len(A), chunk):
        a = A[i:i + chunk]
        d2 = (np.sum(a * a, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
              - 2.0 * (a @ B.T))
        np.maximum(d2, 0.0, out=d2)
        total += float(np.sqrt(d2).sum())
    n_pairs = len(A) * len(B)
    if self_mode:
        n_pairs -= len(A)  # diagonal zeros contribute nothing to `total`
    return total / n_pairs


def energy_distance(X, Y):
    return (2.0 * mean_pairwise_dist(X, Y)
            - mean_pairwise_dist(X) - mean_pairwise_dist(Y))


def rbf_mmd(X, Y, rng):
    """Unbiased MMD^2 (U-statistic) with median-heuristic bandwidth; returns
    (mmd, sigma). Sides above MMD_MAX_N are subsampled deterministically."""
    Xs = X if len(X) <= MMD_MAX_N else X[rng.choice(len(X), MMD_MAX_N, replace=False)]
    Ys = Y if len(Y) <= MMD_MAX_N else Y[rng.choice(len(Y), MMD_MAX_N, replace=False)]
    pool = np.vstack([Xs, Ys])
    med_pool = pool if len(pool) <= MEDIAN_HEURISTIC_N else \
        pool[rng.choice(len(pool), MEDIAN_HEURISTIC_N, replace=False)]
    d2 = (np.sum(med_pool ** 2, axis=1)[:, None]
          + np.sum(med_pool ** 2, axis=1)[None, :]
          - 2.0 * (med_pool @ med_pool.T))
    np.maximum(d2, 0.0, out=d2)
    tri = np.sqrt(d2[np.triu_indices(len(med_pool), k=1)])
    sigma = float(np.median(tri))
    gamma = 1.0 / (2.0 * sigma * sigma)

    def kmean(A, B, exclude_diag):
        d2 = (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
              - 2.0 * (A @ B.T))
        np.maximum(d2, 0.0, out=d2)
        K = np.exp(-gamma * d2)
        if exclude_diag:
            return (K.sum() - np.trace(K)) / (len(A) * (len(A) - 1))
        return K.mean()

    mmd2 = kmean(Xs, Xs, True) + kmean(Ys, Ys, True) - 2.0 * kmean(Xs, Ys, False)
    return float(np.sqrt(max(mmd2, 0.0))), sigma


def in_region_ceiling(X, y):
    """Mean 5-fold CV R^2 of RidgeCV within one region (scaler inside folds)."""
    scores = []
    for tr, te in KFold(CV_FOLDS, shuffle=True, random_state=SEED).split(X):
        sc = StandardScaler().fit(X[tr])
        model = RidgeCV(alphas=RIDGE_CV_ALPHAS).fit(sc.transform(X[tr]), y[tr])
        scores.append(r2_score(y[te], model.predict(sc.transform(X[te]))))
    return float(np.mean(scores))


def region_coefs(df, features, region_col, global_scaler):
    """RidgeCV coefficient vector per region on globally standardized features."""
    coefs = {}
    for r, g in df.groupby(region_col):
        Xg = global_scaler.transform(g[features].values)
        coefs[r] = RidgeCV(alphas=RIDGE_CV_ALPHAS).fit(
            Xg, g[TARGET_COL].values).coef_
    return coefs


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/shift/")
    args = ap.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for pname, data_file, region_col in PARTITIONS:
        df, features = load_data(data_file, None, region_col)
        assert len(features) == 35, "shift measures are defined on the 35 monthly features"
        regions = df[region_col].value_counts().index.tolist()
        print("\n=== partition %s (%d regions, %d rows) ===" %
              (pname, len(regions), len(df)), flush=True)

        global_scaler = StandardScaler().fit(df[features].values)
        coefs = region_coefs(df, features, region_col, global_scaler)
        mat = pd.DataFrame(
            [[cos(coefs[a], coefs[b]) for b in regions] for a in regions],
            index=regions, columns=regions)
        mat.to_csv(out_dir / ("coef_cosine_%s.csv" % pname))
        off_diag = mat.values[~np.eye(len(regions), dtype=bool)]
        print("coefficient cosine matrix -> %s (mean off-diagonal %.3f)"
              % (out_dir / ("coef_cosine_%s.csv" % pname), off_diag.mean()))

        for held_out in regions:
            t0 = time.time()
            train_df = df[df[region_col] != held_out]
            test_df = df[df[region_col] == held_out]
            scaler = StandardScaler().fit(train_df[features].values)
            Xtr = scaler.transform(train_df[features].values)
            Xte = scaler.transform(test_df[features].values)
            ytr = train_df[TARGET_COL].values
            yte = test_df[TARGET_COL].values

            e_dist = energy_distance(Xtr, Xte)
            mmd, sigma = rbf_mmd(Xtr, Xte, np.random.default_rng(SEED))
            wass = float(wasserstein_distance(ytr, yte))
            ceiling = in_region_ceiling(test_df[features].values, yte)
            pool_coef = RidgeCV(alphas=RIDGE_CV_ALPHAS).fit(
                global_scaler.transform(train_df[features].values), ytr).coef_
            ccos = cos(coefs[held_out], pool_coef)

            rows.append({
                "partition": pname, "held_out_region": held_out,
                "n_region": len(test_df), "n_train": len(train_df),
                "energy_dist": e_dist, "mmd_rbf": mmd, "mmd_sigma": sigma,
                "wass_yield": wass, "inregion_r2": ceiling,
                "coef_cos_pool": ccos,
                "yield_mean_region": float(yte.mean()),
                "yield_mean_train": float(ytr.mean()),
            })
            print("  %-22s energy=%6.3f mmd=%6.4f wass=%6.1f "
                  "ceiling=%+.3f coef_cos=%+.3f  (%.1fs)"
                  % (held_out, e_dist, mmd, wass, ceiling, ccos,
                     time.time() - t0), flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "shift_measures.csv", index=False)
    print("\nwrote %s (%d fold rows)" % (out_dir / "shift_measures.csv", len(out)))

    # quick structural check for the write-up: does climate clustering
    # actually reduce fold<->pool covariate distance relative to FRR folds?
    piv = out.groupby("partition")[["energy_dist", "mmd_rbf", "wass_yield",
                                    "inregion_r2"]].mean()
    print("\nper-partition means of the shift measures:")
    print(piv.to_string())


if __name__ == "__main__":
    main()
