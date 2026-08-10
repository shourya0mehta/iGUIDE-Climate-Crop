"""Build Table 1: partition x method macro R^2 (mean +- std over seeds).

Reads the committed per-seed raw results in results/partitions/ — no
model training, no torch import — so a fresh clone can regenerate the paper
table with only pandas/numpy installed:

    python training/make_main_table.py

Writes results/main_table.csv (tidy) and prints the formatted table plus the
per-fold query sizes the macro averages are taken over.
"""
import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARTITIONS = ["state", "ers", "clim"]
FEATURE_SETS = ["seasonal", "monthly"]
METHOD_ORDER = ["ridge", "mlp", "ridge_support_only", "ridge_refit",
                "mlp_finetune", "ccpa_no_climate", "ccpa"]
PARTITION_LABELS = {"state": "state-level FRR", "ers": "county ERS",
                    "clim": "climate clusters"}


def load():
    frames = []
    for p in PARTITIONS:
        for f in FEATURE_SETS:
            path = REPO_ROOT / "results" / "partitions" / ("%s_%s" % (p, f)) \
                / "loco_raw.csv"
            raw = pd.read_csv(path)
            raw["partition"], raw["features"] = p, f
            frames.append(raw)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/main_table.csv")
    args = ap.parse_args(argv)

    raw = load()
    n_seeds = raw["seed"].nunique()
    per_seed = (raw.groupby(["features", "partition", "seed", "method"])["r2"]
                .mean().reset_index())
    tab = (per_seed.groupby(["features", "partition", "method"])["r2"]
           .agg(r2_mean="mean", r2_std="std").reset_index())
    n_folds = (raw.groupby(["features", "partition"])["held_out_region"]
               .nunique().rename("n_folds").reset_index())
    tab = tab.merge(n_folds, on=["features", "partition"])
    tab.to_csv(REPO_ROOT / args.out, index=False)
    print("wrote %s (%d rows; %d seeds)\n" % (args.out, len(tab), n_seeds))

    cells = tab.set_index(["features", "partition", "method"])
    header = ["method"] + ["%s %s" % (f[:4], p)
                           for f in FEATURE_SETS for p in PARTITIONS]
    print("Table 1 — macro R^2, mean ± std over %d seeds "
          "(equal-weight over held-out folds)" % n_seeds)
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for m in METHOD_ORDER:
        row = [("`%s`" % m)]
        for f in FEATURE_SETS:
            for p in PARTITIONS:
                r = cells.loc[(f, p, m)]
                row.append("%+.3f ± %.3f" % (r["r2_mean"], r["r2_std"]))
        print("| " + " | ".join(row) + " |")

    print("\nper-fold query sizes (n_query at support size 32):")
    nq = (raw[raw["features"] == "monthly"]
          .groupby(["partition", "held_out_region"])["n_query"].first())
    for p in PARTITIONS:
        sizes = nq.loc[p].sort_values(ascending=False)
        print("  %-16s %d folds: %s" % (
            PARTITION_LABELS[p], len(sizes),
            ", ".join("%s=%d" % (r, n) for r, n in sizes.items())))


if __name__ == "__main__":
    main()
