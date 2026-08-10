"""Build master_dataset_monthly.csv: per-month climate features, April-October.

Identical pipeline to download_hrrr.py + build_master.py (whose fetch logic and
join/acceptance structure this reuses) with one change: HRRR Daily rows are
aggregated per ('FIPS Code', 'Month') instead of per county-season, then
pivoted wide to 35 feature columns named {variable}_{MM} (7 months x 5
variables, variable-major order). total_precip sums; the other four average.

Only growing-season months (04-10) are fetched — the seasonal build fetched all
12 and discarded 01-03/11-12 after filtering, so the retained data is the same.
Resumable via hrrr_monthly_checkpoint.csv with the same per-state-year pattern.

Acceptance checks mirror build_master.py and run before anything is written:
8691 rows, 42 cols, zero nulls (any missing county-month is reported, never
imputed), exact region counts, Champaign IL 2019 spot check, and fips/year
key-set equality with master_dataset.csv. The seasonal-vs-monthly
reconciliation (mean-of-monthly-means vs mean-of-daily) is reported, not
asserted — they legitimately differ because months have unequal day counts.
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR))

from download_hrrr import fetch_month, STATE_FIPS, YEARS  # noqa: E402
from build_master import REGION_MAP, EXPECTED_REGION_COUNTS, USDA_CSV  # noqa: E402

CHECKPOINT = DATA_DIR / "hrrr_monthly_checkpoint.csv"
MONTHLY_LONG = DATA_DIR / "hrrr_monthly_2017_2022.csv"
MASTER_SEASONAL = DATA_DIR.parent / "master_dataset.csv"
OUTPUT = DATA_DIR.parent / "master_dataset_monthly.csv"

MONTHS = list(range(4, 11))  # April-October inclusive
VARIABLES = ["avg_temp", "total_precip", "avg_humidity", "avg_radiation", "avg_vpd"]
FEATURE_COLS = ["%s_%02d" % (v, m) for v in VARIABLES for m in MONTHS]
ID_HEAD = ["fips", "state_name", "county_name", "year", "yield_bu_acre"]
ID_TAIL = ["state", "region"]
COLUMNS = ID_HEAD + FEATURE_COLS + ID_TAIL


def fetch_state_year_monthly(year, state):
    """Aggregate one state-year per (county, month); None if no months exist."""
    with ThreadPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(lambda m: fetch_month(year, state, m), MONTHS))
    frames = [f for f in frames if f is not None]
    if not frames:
        return None, 0
    combined = pd.concat(frames, ignore_index=True)
    daily = combined[combined["Daily/Monthly"] == "Daily"]
    growing = daily[daily["Month"].between(4, 10)]
    agg = growing.groupby(["FIPS Code", "Month"]).agg(
        avg_temp=("Avg Temperature (K)", "mean"),
        total_precip=("Precipitation (kg m**-2)", "sum"),  # sum — the rest are means
        avg_humidity=("Relative Humidity (%)", "mean"),
        avg_radiation=("Downward Shortwave Radiation Flux (W m**-2)", "mean"),
        avg_vpd=("Vapor Pressure Deficit (kPa)", "mean"),
    ).reset_index()
    agg["year"] = year
    agg["state"] = state
    return agg, len(frames)


def download_monthly():
    frames = []
    done = set()
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        frames.append(ckpt)
        done = set(zip(ckpt["state"], ckpt["year"]))
        print("Resuming: %d state-years already in checkpoint" % len(done), flush=True)

    states = sorted(STATE_FIPS)
    total = len(YEARS) * len(states)
    n_done = len(done)
    t0 = time.time()
    for year in YEARS:
        for state in states:
            if (state, year) in done:
                continue
            agg, n_months = fetch_state_year_monthly(year, state)
            n_done += 1
            if agg is None:
                print("SKIP %s %s: no monthly files upstream" % (state, year), flush=True)
                continue
            frames.append(agg)
            pd.concat(frames, ignore_index=True).to_csv(CHECKPOINT, index=False)
            print("[%d/%d] %s %s: %d county-months from %d months (%.0fs elapsed)"
                  % (n_done, total, state, year, len(agg), n_months, time.time() - t0),
                  flush=True)

    long = pd.concat(frames, ignore_index=True)
    long.to_csv(MONTHLY_LONG, index=False)
    print("wrote %s (%d county-month rows)" % (MONTHLY_LONG, len(long)), flush=True)
    return long


def pivot_wide(long):
    months_per_cy = long.groupby(["FIPS Code", "year"])["Month"].nunique()
    short = months_per_cy[months_per_cy < len(MONTHS)]
    if len(short):
        print("note: %d county-years have <7 months upstream (only a problem if "
              "they survive the yield join):" % len(short))
        print(short.head(20).to_string())

    wide = long.pivot(index=["FIPS Code", "year", "state"], columns="Month",
                      values=VARIABLES)
    wide.columns = ["%s_%02d" % (var, month) for var, month in wide.columns]
    wide = wide.reset_index()[["FIPS Code", "year", "state"] + FEATURE_COLS]
    return wide


def build(wide):
    usda = pd.read_csv(USDA_CSV)
    wide = wide.rename(columns={"FIPS Code": "fips"})
    wide["fips"] = wide["fips"].astype(str).str.zfill(5)
    usda["fips"] = usda["fips"].astype(str).str.zfill(5)
    merged = pd.merge(usda, wide, on=["fips", "year"], how="inner")
    merged["region"] = merged["state"].map(REGION_MAP)
    merged = merged[COLUMNS]

    errors = []
    if merged.shape != (8691, 42):
        errors.append("shape is %s, expected (8691, 42)" % (merged.shape,))
    nulls = merged.isnull().sum()
    if nulls.any():
        bad = merged[merged[FEATURE_COLS].isnull().any(axis=1)]
        for _, row in bad.head(25).iterrows():
            miss = [c for c in FEATURE_COLS if pd.isnull(row[c])]
            errors.append("missing county-months for fips=%s year=%s: %s"
                          % (row["fips"], row["year"], miss))
        errors.append("nulls present: %s" % nulls[nulls > 0].to_dict())
    counts = merged["region"].value_counts().to_dict()
    if counts != EXPECTED_REGION_COUNTS:
        errors.append("region counts %s, expected %s" % (counts, EXPECTED_REGION_COUNTS))
    spot = merged[(merged["fips"] == "17019") & (merged["year"] == 2019)]
    if len(spot) != 1 or spot["yield_bu_acre"].iloc[0] != 180.8:
        errors.append("Champaign IL 2019 spot check failed")

    seasonal = pd.read_csv(MASTER_SEASONAL, dtype={"fips": str})
    keys_monthly = set(zip(merged["fips"], merged["year"]))
    keys_seasonal = set(zip(seasonal["fips"], seasonal["year"]))
    if keys_monthly != keys_seasonal:
        errors.append("fips/year key drift vs master_dataset.csv: %d only-monthly, "
                      "%d only-seasonal"
                      % (len(keys_monthly - keys_seasonal),
                         len(keys_seasonal - keys_monthly)))

    if errors:
        print("MISMATCH — %s NOT written:" % OUTPUT.name)
        for e in errors:
            print("  - %s" % e)
        sys.exit(1)

    merged.to_csv(OUTPUT, index=False)
    print("OK: wrote %s (%d rows, %d cols)" % (OUTPUT, merged.shape[0], merged.shape[1]))
    print("Region counts:")
    print(merged["region"].value_counts().to_string())

    # Reconciliation report (not asserted): monthly-derived seasonal aggregates
    # vs the original season-long dailies. Precip should match to float error;
    # the four means differ slightly (unequal month lengths).
    joined = pd.merge(seasonal, merged, on=["fips", "year"], suffixes=("_s", "_m"))
    print("\nSeasonal-vs-monthly reconciliation (abs diff — mean / max):")
    for var in VARIABLES:
        cols = ["%s_%02d" % (var, m) for m in MONTHS]
        recon = joined[cols].sum(axis=1) if var == "total_precip" \
            else joined[cols].mean(axis=1)
        diff = (recon - joined[var]).abs()
        print("  %-14s %.6g / %.6g" % (var, diff.mean(), diff.max()))


def main():
    if MONTHLY_LONG.exists():
        print("using cached %s" % MONTHLY_LONG)
        long = pd.read_csv(MONTHLY_LONG)
    else:
        long = download_monthly()
    build(pivot_wide(long))


if __name__ == "__main__":
    main()
