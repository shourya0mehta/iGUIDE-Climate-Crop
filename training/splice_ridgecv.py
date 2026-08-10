"""Splice RidgeCV ridge_support_only / ridge_refit into existing raw results.

The RidgeCV fix (see loco_eval.RIDGE_CV_ALPHAS) changes only the two
support-using ridge baselines. Those depend on (data file, seed, fold) alone —
the support/query split comes from `np.random.default_rng(seed)` and ridge
never touches torch state — so they can be recomputed standalone and are
byte-identical to what a full harness re-run would produce. The expensive
torch methods (mlp, mlp_finetune, ccpa_*) are provably unaffected and are
carried over from the stored raw.

Verification before any splice: the FIXED-alpha values of every ridge cell
(including the untouched zero-support `ridge`) are recomputed and asserted
equal to the stored raw values (rtol 1e-9). If that fails, the reproduced
splits are not identical to the original run's and nothing is written.

Usage:
    python training/splice_ridgecv.py --mode loco \
        --data master_dataset_monthly.csv --raw results/monthly/loco_raw.csv \
        --out results/partitions/state_monthly/
    python training/splice_ridgecv.py --mode sweep \
        --data master_dataset_monthly.csv --raw results/sweep_monthly/sweep_raw.csv \
        --out results/sweep_monthly_cv/
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.baselines import TARGET_COL, REGION_COL  # noqa: E402
import training.loco_eval as loco  # noqa: E402
import training.support_sweep as sweep  # noqa: E402

SPLICED = ["ridge_support_only", "ridge_refit"]


def fold_arrays(df, features, held_out, seed, region_col):
    """Reproduce the per-fold preprocessing of loco_eval/support_sweep exactly."""
    train_df = df[df[region_col] != held_out]
    test_df = df[df[region_col] == held_out]
    scaler = StandardScaler().fit(train_df[features].values)
    y_mu = float(train_df[TARGET_COL].mean())
    y_sd = float(train_df[TARGET_COL].std())
    X_train = scaler.transform(train_df[features].values).astype(np.float32)
    y_train_z = ((train_df[TARGET_COL].values - y_mu) / y_sd).astype(np.float32)
    X_test = scaler.transform(test_df[features].values).astype(np.float32)
    y_test_raw = test_df[TARGET_COL].values.astype(np.float64)
    idx = np.random.default_rng(seed).permutation(len(test_df))
    return X_train, y_train_z, X_test, y_test_raw, idx, y_mu, y_sd


def metrics(preds_z, y_qry_raw, y_mu, y_sd):
    preds = np.asarray(preds_z, dtype=np.float64) * y_sd + y_mu
    return (float(r2_score(y_qry_raw, preds)),
            float(np.sqrt(np.mean((y_qry_raw - preds) ** 2))))


def check(stored, computed, what):
    ok = np.isclose(stored, computed, rtol=1e-9, atol=1e-12)
    assert ok, "%s: stored %r != recomputed fixed-alpha %r — split not " \
               "reproduced, refusing to splice" % (what, stored, computed)


def fit_pair(X_sup, y_sup_z, X_train, y_train_z, X_qry, alphas=None):
    """Return predictions for (ridge_support_only, ridge_refit); fixed alpha
    when alphas is None, RidgeCV otherwise. Also returns chosen alphas."""
    Xr = np.vstack([X_train, X_sup])
    yr = np.concatenate([y_train_z, y_sup_z])
    if alphas is None:
        m_sup = Ridge(alpha=loco.RIDGE_ALPHA).fit(X_sup, y_sup_z)
        m_ref = Ridge(alpha=loco.RIDGE_ALPHA).fit(Xr, yr)
        a_sup = a_ref = loco.RIDGE_ALPHA
    else:
        m_sup = RidgeCV(alphas=alphas).fit(X_sup, y_sup_z)
        m_ref = RidgeCV(alphas=alphas).fit(Xr, yr)
        a_sup, a_ref = float(m_sup.alpha_), float(m_ref.alpha_)
    return m_sup.predict(X_qry), m_ref.predict(X_qry), a_sup, a_ref


def splice_loco(df, features, raw, region_col):
    region_order = df[region_col].value_counts().index.tolist()
    seeds = sorted(raw["seed"].unique())
    raw = raw.set_index(["seed", "held_out_region", "method"])
    chosen = []
    for seed in seeds:
        for held_out in region_order:
            (X_train, y_train_z, X_test, y_test_raw,
             idx, y_mu, y_sd) = fold_arrays(df, features, held_out, seed, region_col)
            sup_idx, qry_idx = idx[:loco.SUPPORT_SIZE], idx[loco.SUPPORT_SIZE:]
            X_sup, X_qry = X_test[sup_idx], X_test[qry_idx]
            y_sup_z = ((y_test_raw[sup_idx] - y_mu) / y_sd).astype(np.float32)
            y_qry = y_test_raw[qry_idx]

            # --- verify split reproduction against every stored fixed-alpha cell
            r2_plain, _ = metrics(Ridge(alpha=loco.RIDGE_ALPHA)
                                  .fit(X_train, y_train_z).predict(X_qry),
                                  y_qry, y_mu, y_sd)
            check(raw.loc[(seed, held_out, "ridge"), "r2"], r2_plain,
                  "seed %d %s ridge" % (seed, held_out))
            p_sup, p_ref, _, _ = fit_pair(X_sup, y_sup_z, X_train, y_train_z, X_qry)
            for m, p in zip(SPLICED, (p_sup, p_ref)):
                check(raw.loc[(seed, held_out, m), "r2"],
                      metrics(p, y_qry, y_mu, y_sd)[0],
                      "seed %d %s %s" % (seed, held_out, m))

            # --- recompute under RidgeCV and splice
            p_sup, p_ref, a_sup, a_ref = fit_pair(
                X_sup, y_sup_z, X_train, y_train_z, X_qry, loco.RIDGE_CV_ALPHAS)
            for m, p, a in zip(SPLICED, (p_sup, p_ref), (a_sup, a_ref)):
                r2, rmse = metrics(p, y_qry, y_mu, y_sd)
                raw.loc[(seed, held_out, m), ["r2", "rmse"]] = (r2, rmse)
                chosen.append({"seed": seed, "held_out_region": held_out,
                               "method": m, "support_size": loco.SUPPORT_SIZE,
                               "alpha": a})
    return raw.reset_index(), pd.DataFrame(chosen), region_order, seeds


def splice_sweep(df, features, raw, region_col):
    region_order = df[region_col].value_counts().index.tolist()
    seeds = sorted(raw["seed"].unique())
    raw = raw.set_index(["seed", "held_out_region", "method", "support_size"])
    chosen = []
    for seed in seeds:
        for held_out in region_order:
            (X_train, y_train_z, X_test, y_test_raw,
             idx, y_mu, y_sd) = fold_arrays(df, features, held_out, seed, region_col)
            qry_idx = idx[sweep.MAX_SUPPORT:]
            X_qry, y_qry = X_test[qry_idx], y_test_raw[qry_idx]

            r2_plain, _ = metrics(Ridge(alpha=loco.RIDGE_ALPHA)
                                  .fit(X_train, y_train_z).predict(X_qry),
                                  y_qry, y_mu, y_sd)
            check(raw.loc[(seed, held_out, "ridge", 0), "r2"], r2_plain,
                  "seed %d %s ridge s=0" % (seed, held_out))

            for s in sweep.SUPPORT_SIZES:
                sup_idx = idx[:s]
                X_sup = X_test[sup_idx]
                y_sup_z = ((y_test_raw[sup_idx] - y_mu) / y_sd).astype(np.float32)

                p_sup, p_ref, _, _ = fit_pair(X_sup, y_sup_z, X_train,
                                              y_train_z, X_qry)
                for m, p in zip(SPLICED, (p_sup, p_ref)):
                    check(raw.loc[(seed, held_out, m, s), "r2"],
                          metrics(p, y_qry, y_mu, y_sd)[0],
                          "seed %d %s %s s=%d" % (seed, held_out, m, s))

                p_sup, p_ref, a_sup, a_ref = fit_pair(
                    X_sup, y_sup_z, X_train, y_train_z, X_qry,
                    loco.RIDGE_CV_ALPHAS)
                for m, p, a in zip(SPLICED, (p_sup, p_ref), (a_sup, a_ref)):
                    r2, rmse = metrics(p, y_qry, y_mu, y_sd)
                    raw.loc[(seed, held_out, m, s), ["r2", "rmse"]] = (r2, rmse)
                    chosen.append({"seed": seed, "held_out_region": held_out,
                                   "method": m, "support_size": s, "alpha": a})
    return raw.reset_index(), pd.DataFrame(chosen), region_order, seeds


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["loco", "sweep"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw", required=True, help="existing fixed-alpha raw csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--region-col", default=REGION_COL)
    args = ap.parse_args(argv)

    df, features = loco.load_data(args.data, None, args.region_col)
    raw = pd.read_csv(args.raw)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "loco":
        patched, chosen, region_order, seeds = splice_loco(
            df, features, raw, args.region_col)
        summary, per_seed = loco.summarize(patched, region_order)
        patched.to_csv(out_dir / "loco_raw.csv", index=False)
        summary.to_csv(out_dir / "loco_summary.csv", index=False)
        print("verified %d fixed-alpha cells; spliced %d RidgeCV cells"
              % (3 * len(seeds) * len(region_order), len(chosen)))
        loco.print_summary(summary, per_seed, region_order, n_seeds=len(seeds))
    else:
        patched, chosen, region_order, seeds = splice_sweep(
            df, features, raw, args.region_col)
        summary, per_seed = sweep.summarize(patched, region_order)
        patched.to_csv(out_dir / "sweep_raw.csv", index=False)
        summary.to_csv(out_dir / "sweep_summary.csv", index=False)
        print("verified %d fixed-alpha cells; spliced %d RidgeCV cells"
              % ((1 + 2 * len(sweep.SUPPORT_SIZES)) * len(seeds) * len(region_order),
                 len(chosen)))
        sweep.print_summary(summary, per_seed, region_order, n_seeds=len(seeds))

    chosen.to_csv(out_dir / "ridgecv_alphas.csv", index=False)
    print("\nchosen alpha by method x support size (median [min, max]):")
    g = chosen.groupby(["method", "support_size"])["alpha"]
    for (m, s), vals in g:
        print("  %-18s s=%-3d median %8.2f  [%g, %g]"
              % (m, s, vals.median(), vals.min(), vals.max()))


if __name__ == "__main__":
    main()
