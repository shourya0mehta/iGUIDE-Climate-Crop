"""Corrected leave-one-cluster-out (LOCO) evaluation.

Fixes the three bugs that invalidated the original notebook evaluation:

1. Meta-training leakage   -> a separate model is trained per fold; fold k
   meta-trains only on the five regions that are not k.
2. Normalization leakage   -> the feature scaler and the target mean/std are
   fit on the training regions only, inside each fold.
3. Support-set asymmetry   -> one support/query split per (seed, fold), shared
   by every method (asserted), with baselines that also use the support set.

Usage:
    python training/loco_eval.py --seeds 0 1 2 3 4 --data master_dataset.csv --out results/
"""
import argparse
import copy
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.baselines import TARGET_COL, REGION_COL, MLP  # noqa: E402
from models.ccpa import CCPA, CCPANoClimate  # noqa: E402
from training.maml_trainer import meta_train, adapt_and_predict  # noqa: E402

# Protocol constants (from the experiment spec — change deliberately, not casually)
SUPPORT_SIZE = 32
QUERY_SIZE = 32          # meta-training episodes only; test query = rest of region
INNER_LR = 1e-3
OUTER_LR = 3e-4
INNER_STEPS_TRAIN = 3
INNER_STEPS_TEST = 10
META_EPOCHS = 200
GRAD_CLIP = 1.0
RIDGE_ALPHA = 1.0
MLP_EPOCHS = 100
MLP_LR = 1e-3
MLP_BATCH = 64
FINETUNE_STEPS = 10      # mlp_finetune uses the same SGD(lr=INNER_LR) budget as
                         # the MAML meta-test inner loop, so the only difference
                         # vs ccpa_* is the initialization.

METHOD_ORDER = ["ridge", "mlp", "ridge_support_only", "ridge_refit",
                "mlp_finetune", "ccpa_no_climate", "ccpa"]

# Feature columns are inferred: everything not in this list. Overridable
# with --features for ad-hoc subsets.
NON_FEATURE_COLS = ["fips", "state_name", "county_name", "year",
                    "yield_bu_acre", "state", "region"]

DISPLAY_NAMES = {
    "Heartland": "Heartland",
    "Northern Crescent": "N.Crescent",
    "Northern Great Plains": "N.Gt.Plains",
    "Prairie Gateway": "Pr.Gateway",
    "Southern Seaboard": "S.Seaboard",
    "Basin and Range": "Basin&Range",
    "MACRO": "MACRO",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data(path, features_arg=None):
    """Load the dataset and return (df, feature_cols).

    Features default to every column not in NON_FEATURE_COLS, in file order;
    pass a comma-separated `features_arg` to override.
    """
    p = Path(path)
    if not p.exists():
        sys.exit("error: data file not found: %s\n"
                 "Expected a master dataset with '%s' and '%s' columns."
                 % (p.resolve(), TARGET_COL, REGION_COL))
    df = pd.read_csv(p)
    missing = [c for c in (TARGET_COL, REGION_COL) if c not in df.columns]
    if missing:
        sys.exit("error: data file is missing columns: %s" % missing)
    if features_arg:
        features = [c.strip() for c in features_arg.split(",") if c.strip()]
        missing = [c for c in features if c not in df.columns]
        if missing:
            sys.exit("error: --features not present in data: %s" % missing)
    else:
        features = [c for c in df.columns if c not in NON_FEATURE_COLS]
    if not features:
        sys.exit("error: no feature columns inferred from %s" % list(df.columns))
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        sys.exit("error: non-numeric feature columns: %s" % non_numeric)
    needed = features + [TARGET_COL, REGION_COL]
    if df[needed].isna().any().any():
        sys.exit("error: nulls found in feature/target/region columns")
    counts = df[REGION_COL].value_counts()
    if len(counts) != 6:
        sys.exit("error: expected 6 regions, found %d: %s"
                 % (len(counts), list(counts.index)))
    if counts.min() <= SUPPORT_SIZE:
        sys.exit("error: smallest region (n=%d) does not exceed support size %d"
                 % (counts.min(), SUPPORT_SIZE))
    return df, features


def predict_torch(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).reshape(-1).numpy()


def train_mlp(X, y, seed):
    """Plain MLP fit on the training regions (same recipe as models/baselines.py:
    Adam lr=1e-3, MSE, batch 64, 100 epochs), on pre-scaled X and z-normed y."""
    model = MLP(input_dim=X.shape[1])
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                        batch_size=MLP_BATCH, shuffle=True, generator=gen)
    opt = torch.optim.Adam(model.parameters(), lr=MLP_LR)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(MLP_EPOCHS):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).reshape(-1), yb.reshape(-1)).backward()
            opt.step()
    return model


def finetune_mlp(model, support_X, support_y):
    """FINETUNE_STEPS full-batch SGD(lr=INNER_LR) steps on the support set,
    starting from the train-region MLP — matches the MAML meta-test inner loop."""
    tuned = copy.deepcopy(model)
    opt = torch.optim.SGD(tuned.parameters(), lr=INNER_LR)
    loss_fn = nn.MSELoss()
    sx = torch.from_numpy(support_X)
    sy = torch.from_numpy(support_y)
    tuned.train()
    for _ in range(FINETUNE_STEPS):
        opt.zero_grad()
        loss_fn(tuned(sx).reshape(-1), sy).backward()
        opt.step()
    return tuned


def run_fold(df, features, held_out, fold_idx, seed, meta_epochs):
    """Evaluate all seven methods on one (seed, held-out region) fold.

    Returns a list of result-row dicts (one per method).
    """
    t0 = time.time()
    set_seed(seed)

    train_df = df[df[REGION_COL] != held_out]
    test_df = df[df[REGION_COL] == held_out]

    # --- Bug-1 guard: fold k trains on exactly the five regions that are not k
    train_regions = set(train_df[REGION_COL].unique())
    assert len(train_regions) == 5 and held_out not in train_regions, \
        "fold %s: bad training-region set %s" % (held_out, train_regions)

    # --- Bug-2 guard: scaler and target stats fit on training regions only
    scaler = StandardScaler().fit(train_df[features].values)
    assert int(np.asarray(scaler.n_samples_seen_).ravel()[0]) == len(train_df), \
        "scaler saw %s samples, expected %d" % (scaler.n_samples_seen_, len(train_df))
    y_mu = float(train_df[TARGET_COL].mean())
    y_sd = float(train_df[TARGET_COL].std())

    # --- One support/query split per (seed, fold), shared by every method
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(test_df))
    support_idx = idx[:SUPPORT_SIZE]
    query_idx = idx[SUPPORT_SIZE:]
    split_checksum = hashlib.sha1(query_idx.tobytes()).hexdigest()

    X_train = scaler.transform(train_df[features].values).astype(np.float32)
    y_train_z = ((train_df[TARGET_COL].values - y_mu) / y_sd).astype(np.float32)
    X_test = scaler.transform(test_df[features].values).astype(np.float32)
    y_test_raw = test_df[TARGET_COL].values.astype(np.float64)

    X_sup = X_test[support_idx]
    y_sup_z = ((y_test_raw[support_idx] - y_mu) / y_sd).astype(np.float32)
    X_qry = X_test[query_idx]
    y_qry_raw = y_test_raw[query_idx]

    print("[seed %d | fold %d/6 | held-out %s] n_train=%d n_support=%d n_query=%d"
          % (seed, fold_idx + 1, held_out, len(train_df), len(support_idx),
             len(query_idx)), flush=True)

    rows = []
    used_checksums = {}

    def score(method, preds_z, q_idx):
        # --- Bug-3 guard bookkeeping: record the query indices each method
        # was actually scored on; asserted identical below.
        used_checksums[method] = hashlib.sha1(q_idx.tobytes()).hexdigest()
        preds = np.asarray(preds_z, dtype=np.float64) * y_sd + y_mu  # bu/acre
        r2 = float(r2_score(y_qry_raw, preds))
        rmse = float(np.sqrt(np.mean((y_qry_raw - preds) ** 2)))
        rows.append({"seed": seed, "held_out_region": held_out, "method": method,
                     "r2": r2, "rmse": rmse, "n_query": len(q_idx)})
        print("    %-18s R2=%7.3f  RMSE=%7.2f" % (method, r2, rmse), flush=True)

    # 1. ridge: train regions only
    ridge = Ridge(alpha=RIDGE_ALPHA).fit(X_train, y_train_z)
    score("ridge", ridge.predict(X_qry), query_idx)

    # 2. mlp: train regions only
    mlp = train_mlp(X_train, y_train_z, seed)
    score("mlp", predict_torch(mlp, X_qry), query_idx)

    # 3. ridge_support_only: the 32 support rows only
    ridge_sup = Ridge(alpha=RIDGE_ALPHA).fit(X_sup, y_sup_z)
    score("ridge_support_only", ridge_sup.predict(X_qry), query_idx)

    # 4. ridge_refit: train regions + support rows
    ridge_refit = Ridge(alpha=RIDGE_ALPHA).fit(
        np.vstack([X_train, X_sup]), np.concatenate([y_train_z, y_sup_z]))
    score("ridge_refit", ridge_refit.predict(X_qry), query_idx)

    # 5. mlp_finetune: the trained MLP + 10 SGD steps on support
    mlp_tuned = finetune_mlp(mlp, X_sup, y_sup_z)
    score("mlp_finetune", predict_torch(mlp_tuned, X_qry), query_idx)

    # 6-7. MAML-trained CCPA ablation and full CCPA
    region_tensors = {}
    for r, g in train_df.groupby(REGION_COL):
        Xr = scaler.transform(g[features].values).astype(np.float32)
        yr = ((g[TARGET_COL].values - y_mu) / y_sd).astype(np.float32)
        region_tensors[r] = (torch.from_numpy(Xr), torch.from_numpy(yr))
    maml_regions = set(region_tensors)
    assert len(maml_regions) == 5 and held_out not in maml_regions, \
        "MAML training set for fold %s is wrong: %s" % (held_out, maml_regions)

    for method, model_cls in [("ccpa_no_climate", CCPANoClimate), ("ccpa", CCPA)]:
        set_seed(seed)  # same init stream for both models
        model = model_cls(input_dim=len(features))
        maml = meta_train(
            model, region_tensors, epochs=meta_epochs,
            inner_lr=INNER_LR, outer_lr=OUTER_LR,
            support_size=SUPPORT_SIZE, query_size=QUERY_SIZE,
            inner_steps=INNER_STEPS_TRAIN, grad_clip=GRAD_CLIP,
            first_order=True,
            rng=np.random.default_rng([seed, fold_idx]),  # same episodes for both
            label="%s:" % method)
        preds = adapt_and_predict(maml, X_sup, y_sup_z, X_qry,
                                  inner_steps=INNER_STEPS_TEST)
        score(method, preds, query_idx)

    # --- Bug-3 guard: all seven methods scored on identical query indices
    assert set(used_checksums) == set(METHOD_ORDER), \
        "methods evaluated: %s" % sorted(used_checksums)
    assert set(used_checksums.values()) == {split_checksum}, \
        "methods scored on different query splits: %s" % used_checksums

    print("    fold done in %.1fs" % (time.time() - t0), flush=True)
    return rows


def summarize(raw, region_order):
    grp = raw.groupby(["held_out_region", "method"])
    assert grp["n_query"].nunique().eq(1).all(), \
        "n_query varies across seeds within a fold"
    per_region = grp.agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        n_query=("n_query", "first")).reset_index()

    # MACRO: equal-weight mean over the six regions within each seed,
    # then mean/std across seeds. n_query is the total across regions.
    per_seed = raw.groupby(["seed", "method"]).agg(
        r2=("r2", "mean"), rmse=("rmse", "mean"),
        n_query=("n_query", "sum")).reset_index()
    macro = per_seed.groupby("method").agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        n_query=("n_query", "first")).reset_index()
    macro.insert(0, "held_out_region", "MACRO")

    summary = pd.concat([per_region, macro], ignore_index=True)
    summary["held_out_region"] = pd.Categorical(
        summary["held_out_region"], categories=region_order + ["MACRO"], ordered=True)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True)
    summary = summary.sort_values(["held_out_region", "method"]).reset_index(drop=True)
    return summary, per_seed


def print_summary(summary, per_seed, region_order, n_seeds):
    cols = region_order + ["MACRO"]
    disp = [DISPLAY_NAMES.get(c, c) for c in cols]

    nq = summary[summary["method"] == "ridge"].set_index("held_out_region")["n_query"]
    print("\nn_query per held-out region (MACRO = total):")
    print("  " + "  ".join("%s=%d" % (DISPLAY_NAMES.get(c, c), nq.loc[c]) for c in cols))

    for metric, title in [("r2", "R^2"), ("rmse", "RMSE (bu/acre)")]:
        cells = [
            "%.3f" % m if pd.isna(s) else "%.3f ± %.3f" % (m, s)
            for m, s in zip(summary[metric + "_mean"], summary[metric + "_std"])
        ]
        pivot = (summary.assign(cell=cells)
                 .pivot(index="method", columns="held_out_region", values="cell")
                 .reindex(METHOD_ORDER)[cols])
        pivot.columns = disp
        pivot.index.name = None
        print("\n=== %s by held-out region (mean ± std over %d seed%s) ==="
              % (title, n_seeds, "" if n_seeds == 1 else "s"))
        print(pivot.to_string())

    mac = summary[summary["held_out_region"] == "MACRO"].set_index("method")
    print("\n=== Key comparison (MACRO R^2, equal-weight across regions) ===")
    for m in ["ccpa", "ccpa_no_climate", "mlp_finetune", "ridge_refit"]:
        std = mac.loc[m, "r2_std"]
        print("  %-18s %.3f%s" % (m, mac.loc[m, "r2_mean"],
                                  "" if pd.isna(std) else " ± %.3f" % std))
    wide = per_seed.pivot(index="seed", columns="method", values="r2")
    for rival in ["mlp_finetune", "ridge_refit"]:
        wins = int((wide["ccpa"] > wide[rival]).sum())
        beats = mac.loc["ccpa", "r2_mean"] > mac.loc[rival, "r2_mean"]
        print("  ccpa %s %-13s on MACRO R^2 mean; per-seed wins %d/%d"
              % ("beats" if beats else "does NOT beat", rival, wins, len(wide)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="master_dataset.csv")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--meta-epochs", type=int, default=META_EPOCHS,
                    help="lower only for quick plumbing checks (default %d)" % META_EPOCHS)
    ap.add_argument("--features", default=None,
                    help="comma-separated feature columns; default: every column "
                         "not in %s" % NON_FEATURE_COLS)
    args = ap.parse_args(argv)

    df, features = load_data(args.data, args.features)
    print("features (%d): %s" % (len(features),
          ", ".join(features) if len(features) <= 8
          else ", ".join(features[:6]) + ", ... +%d more" % (len(features) - 6)))
    region_order = df[REGION_COL].value_counts().index.tolist()  # largest first
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rows = []
    for seed in args.seeds:
        for fold_idx, held_out in enumerate(region_order):
            rows.extend(run_fold(df, features, held_out, fold_idx, seed,
                                 args.meta_epochs))

    raw = pd.DataFrame(rows, columns=["seed", "held_out_region", "method",
                                      "r2", "rmse", "n_query"])
    summary, per_seed = summarize(raw, region_order)

    raw_path = out_dir / "loco_raw.csv"
    summary_path = out_dir / "loco_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nwrote %s (%d rows) and %s (%d rows) in %.1f min total"
          % (raw_path, len(raw), summary_path, len(summary), (time.time() - t0) / 60))
    print_summary(summary, per_seed, region_order, n_seeds=len(args.seeds))


if __name__ == "__main__":
    main()
