"""Support-size sweep {8, 16, 32, 64} on the corrected LOCO protocol.

Tests whether meta-learning's advantage widens as target-region labels shrink.
Protocol, leakage guards, and method definitions are inherited unchanged from
training/loco_eval.py; only the support size varies.

The one extra invariant: within a (seed, fold) the QUERY SET IS FIXED to the
rows excluded at the LARGEST support size (64), so R^2 is comparable across
sweep points — support sets are nested prefixes of one permutation and every
(method, support_size) scores on identical query rows (asserted).

Zero-support references `ridge` and `mlp` do not vary with support size; they
are computed once per fold on the same fixed query set and recorded with
support_size=0 for use as reference lines in the figure.

Usage:
    python training/support_sweep.py --seeds 0 1 2 3 4 \
        --data master_dataset.csv --out results/sweep/
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.baselines import TARGET_COL, REGION_COL  # noqa: E402
from models.ccpa import CCPA, CCPANoClimate  # noqa: E402
from training.loco_eval import (  # noqa: E402
    DISPLAY_NAMES, GRAD_CLIP, INNER_LR, INNER_STEPS_TRAIN, INNER_STEPS_TEST,
    META_EPOCHS, OUTER_LR, QUERY_SIZE, RIDGE_ALPHA, RIDGE_CV_ALPHAS,
    finetune_mlp, load_data, predict_torch, set_seed, train_mlp)
from training.maml_trainer import meta_train, adapt_and_predict  # noqa: E402

SUPPORT_SIZES = [8, 16, 32, 64]
MAX_SUPPORT = max(SUPPORT_SIZES)
REF_METHODS = ["ridge", "mlp"]                       # support_size=0 rows
SWEEP_METHODS = ["ridge_support_only", "ridge_refit", "mlp_finetune",
                 "ccpa_no_climate", "ccpa"]
METHOD_ORDER = REF_METHODS + SWEEP_METHODS


def run_fold_sweep(df, features, held_out, fold_idx, seed, meta_epochs):
    t0 = time.time()
    set_seed(seed)

    train_df = df[df[REGION_COL] != held_out]
    test_df = df[df[REGION_COL] == held_out]

    n_regions = df[REGION_COL].nunique()
    train_regions = set(train_df[REGION_COL].unique())
    assert len(train_regions) == n_regions - 1 and held_out not in train_regions, \
        "fold %s: bad training-region set %s" % (held_out, train_regions)

    scaler = StandardScaler().fit(train_df[features].values)
    assert int(np.asarray(scaler.n_samples_seen_).ravel()[0]) == len(train_df), \
        "scaler saw %s samples, expected %d" % (scaler.n_samples_seen_, len(train_df))
    y_mu = float(train_df[TARGET_COL].mean())
    y_sd = float(train_df[TARGET_COL].std())

    # One permutation per (seed, fold); support sets are nested prefixes and
    # the query set is fixed to everything after MAX_SUPPORT.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test_df))
    query_idx = idx[MAX_SUPPORT:]
    split_checksum = hashlib.sha1(query_idx.tobytes()).hexdigest()

    X_train = scaler.transform(train_df[features].values).astype(np.float32)
    y_train_z = ((train_df[TARGET_COL].values - y_mu) / y_sd).astype(np.float32)
    X_test = scaler.transform(test_df[features].values).astype(np.float32)
    y_test_raw = test_df[TARGET_COL].values.astype(np.float64)
    X_qry = X_test[query_idx]
    y_qry_raw = y_test_raw[query_idx]

    print("[seed %d | fold %d/%d | held-out %s] n_train=%d n_query=%d (fixed)"
          % (seed, fold_idx + 1, n_regions, held_out, len(train_df),
             len(query_idx)), flush=True)

    rows = []
    used_checksums = {}

    def score(method, support_size, preds_z):
        used_checksums[(method, support_size)] = \
            hashlib.sha1(query_idx.tobytes()).hexdigest()
        preds = np.asarray(preds_z, dtype=np.float64) * y_sd + y_mu
        r2 = float(r2_score(y_qry_raw, preds))
        rmse = float(np.sqrt(np.mean((y_qry_raw - preds) ** 2)))
        rows.append({"seed": seed, "held_out_region": held_out, "method": method,
                     "support_size": support_size, "r2": r2, "rmse": rmse,
                     "n_query": len(query_idx)})
        print("    s=%-2d %-18s R2=%7.3f  RMSE=%7.2f"
              % (support_size, method, r2, rmse), flush=True)

    # Zero-support references, once per fold, on the same fixed query set
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(X_train, y_train_z)
    score("ridge", 0, ridge.predict(X_qry))
    mlp = train_mlp(X_train, y_train_z, seed)
    score("mlp", 0, predict_torch(mlp, X_qry))

    region_tensors = {}
    for r, g in train_df.groupby(REGION_COL):
        Xr = scaler.transform(g[features].values).astype(np.float32)
        yr = ((g[TARGET_COL].values - y_mu) / y_sd).astype(np.float32)
        region_tensors[r] = (torch.from_numpy(Xr), torch.from_numpy(yr))
    maml_regions = set(region_tensors)
    assert len(maml_regions) == n_regions - 1 and held_out not in maml_regions, \
        "MAML training set for fold %s is wrong: %s" % (held_out, maml_regions)

    for s in SUPPORT_SIZES:
        support_idx = idx[:s]
        X_sup = X_test[support_idx]
        y_sup_z = ((y_test_raw[support_idx] - y_mu) / y_sd).astype(np.float32)

        # alpha chosen by internal efficient-LOO CV on the fitting rows only
        # (never on query); at s=8 this is the LOO CV the small-n case needs
        score("ridge_support_only", s,
              RidgeCV(alphas=RIDGE_CV_ALPHAS).fit(X_sup, y_sup_z).predict(X_qry))
        score("ridge_refit", s,
              RidgeCV(alphas=RIDGE_CV_ALPHAS).fit(
                  np.vstack([X_train, X_sup]),
                  np.concatenate([y_train_z, y_sup_z])).predict(X_qry))
        score("mlp_finetune", s,
              predict_torch(finetune_mlp(mlp, X_sup, y_sup_z), X_qry))

        # Meta-training support size tracks the eval support size
        for method, model_cls in [("ccpa_no_climate", CCPANoClimate),
                                  ("ccpa", CCPA)]:
            set_seed(seed)
            model = model_cls(input_dim=len(features))
            maml = meta_train(
                model, region_tensors, epochs=meta_epochs,
                inner_lr=INNER_LR, outer_lr=OUTER_LR,
                support_size=s, query_size=QUERY_SIZE,
                inner_steps=INNER_STEPS_TRAIN, grad_clip=GRAD_CLIP,
                first_order=True,
                rng=np.random.default_rng([seed, fold_idx, s]),
                label="s=%d %s:" % (s, method), log_every=0)
            score(method, s,
                  adapt_and_predict(maml, X_sup, y_sup_z, X_qry,
                                    inner_steps=INNER_STEPS_TEST))

    expected = {(m, 0) for m in REF_METHODS} | \
               {(m, s) for m in SWEEP_METHODS for s in SUPPORT_SIZES}
    assert set(used_checksums) == expected, \
        "missing sweep cells: %s" % sorted(expected - set(used_checksums))
    assert set(used_checksums.values()) == {split_checksum}, \
        "query set varied across sweep cells"

    print("    fold done in %.1fs" % (time.time() - t0), flush=True)
    return rows


def summarize(raw, region_order):
    grp = raw.groupby(["held_out_region", "method", "support_size"])
    assert grp["n_query"].nunique().eq(1).all()
    per_region = grp.agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        n_query=("n_query", "first")).reset_index()

    per_seed = raw.groupby(["seed", "method", "support_size"]).agg(
        r2=("r2", "mean"), rmse=("rmse", "mean"),
        n_query=("n_query", "sum")).reset_index()
    macro = per_seed.groupby(["method", "support_size"]).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        n_query=("n_query", "first")).reset_index()
    macro.insert(0, "held_out_region", "MACRO")

    summary = pd.concat([per_region, macro], ignore_index=True)
    summary["held_out_region"] = pd.Categorical(
        summary["held_out_region"], categories=region_order + ["MACRO"], ordered=True)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(
        ["held_out_region", "method", "support_size"]).reset_index(drop=True)
    return summary, per_seed


def print_summary(summary, per_seed, region_order, n_seeds):
    nq = (summary[summary["method"] == "ridge"]
          .set_index("held_out_region")["n_query"])
    print("\nn_query per held-out region — fixed across all support sizes "
          "(MACRO = total):")
    print("  " + "  ".join("%s=%d" % (DISPLAY_NAMES.get(c, c), nq.loc[c])
                           for c in region_order + ["MACRO"]))
    if "Basin and Range" in region_order:
        print("  (Basin & Range n_query=35: this region's sweep is noisy)")

    mac = summary[summary["held_out_region"] == "MACRO"]
    cells = [
        "%.3f" % m if pd.isna(s) else "%.3f ± %.3f" % (m, s)
        for m, s in zip(mac["r2_mean"], mac["r2_std"])
    ]
    pivot = (mac.assign(cell=cells)
             .pivot(index="method", columns="support_size", values="cell")
             .reindex(METHOD_ORDER))
    pivot.columns = ["s=%d" % c if c else "s=0 (ref)" for c in pivot.columns]
    pivot.index.name = None
    print("\n=== MACRO R^2 by support size (mean ± std over %d seed%s) ==="
          % (n_seeds, "" if n_seeds == 1 else "s"))
    print(pivot.to_string(na_rep=""))

    wide = per_seed.pivot_table(index=["seed", "support_size"],
                                columns="method", values="r2").reset_index()
    print("\n=== ccpa vs mlp_finetune by support size ===")
    for s in SUPPORT_SIZES:
        w = wide[wide["support_size"] == s]
        diff = (w["ccpa"] - w["mlp_finetune"])
        print("  s=%-2d  mean delta %+0.3f  per-seed wins %d/%d"
              % (s, diff.mean(), int((diff > 0).sum()), len(w)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="master_dataset.csv")
    ap.add_argument("--out", default="results/sweep/")
    ap.add_argument("--meta-epochs", type=int, default=META_EPOCHS)
    ap.add_argument("--features", default=None)
    args = ap.parse_args(argv)

    df, features = load_data(args.data, args.features)
    region_order = df[REGION_COL].value_counts().index.tolist()
    smallest = df[REGION_COL].value_counts().min()
    assert smallest > MAX_SUPPORT, \
        "smallest region n=%d must exceed max support %d" % (smallest, MAX_SUPPORT)
    print("features (%d) | support sizes %s | query fixed after s=%d"
          % (len(features), SUPPORT_SIZES, MAX_SUPPORT))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for seed in args.seeds:
        for fold_idx, held_out in enumerate(region_order):
            rows.extend(run_fold_sweep(df, features, held_out, fold_idx, seed,
                                       args.meta_epochs))

    raw = pd.DataFrame(rows, columns=["seed", "held_out_region", "method",
                                      "support_size", "r2", "rmse", "n_query"])
    summary, per_seed = summarize(raw, region_order)
    raw.to_csv(out_dir / "sweep_raw.csv", index=False)
    summary.to_csv(out_dir / "sweep_summary.csv", index=False)
    print("\nwrote %s and %s in %.1f min"
          % (out_dir / "sweep_raw.csv", out_dir / "sweep_summary.csv",
             (time.time() - t0) / 60))
    print_summary(summary, per_seed, region_order, n_seeds=len(args.seeds))


if __name__ == "__main__":
    main()
