"""Build the three spatial-partition datasets for the sensitivity experiment.

Partition 1 (exists): `region` — state-level FRR approximation in
    master_dataset{,_monthly}.csv. Untouched.
Partition 2: `region_ers` — true county-level USDA ERS Farm Resource Regions
    from data/reglink.xls ("Aggregating counties to ERS resource regions",
    USDA/ERS, downloaded 2026-08-10 from
    https://www.ers.usda.gov/sites/default/files/images/reglink.xls
    sha256 2f325589e5f2aef28c98774773007da1253b717c51dde2952f7b014aee22fb37).
    Regions with fewer than MIN_COUNTY_YEARS county-years are dropped (a
    held-out fold that small cannot support a 32-row support set plus a
    meaningful query set); the drop is reported loudly below.
Partition 3: `region_clim` — k-means climate clusters, k = number of ERS
    regions surviving the inclusion rule, so fold counts are like-for-like.
    Clustered on county-level climate normals: each county's 35 monthly
    features averaged across its observed years, standardized across counties.
    Every county-year inherits its county's cluster (a county never changes
    cluster between years, so this is a spatial partition). k-means sees X
    only, never yield.

Data columns are carried through as raw strings so the feature/target values
in the derived CSVs are byte-identical to the source files; only the region
column is replaced.

Outputs: master_dataset_ers.csv, master_dataset_monthly_ers.csv,
         master_dataset_clim.csv, master_dataset_monthly_clim.csv

Usage:
    python data/build_partitions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_COUNTY_YEARS = 150
KMEANS_SEED = 0
KMEANS_N_INIT = 10

ERS_KEY = {1: "Heartland", 2: "Northern Crescent", 3: "Northern Great Plains",
           4: "Prairie Gateway", 5: "Eastern Uplands", 6: "Southern Seaboard",
           7: "Fruitful Rim", 8: "Basin and Range", 9: "Mississippi Portal"}

SPOT_CHECKS = [(17019, "Heartland"),           # Champaign County, IL
               (28011, "Mississippi Portal"),  # Bolivar County, MS
               (51195, "Eastern Uplands")]     # Wise County, VA


def load_crosswalk():
    raw = pd.read_excel(REPO_ROOT / "data" / "reglink.xls", header=None)
    header_row = raw.index[raw[0].astype(str).str.strip() == "Fips"]
    assert len(header_row) == 1, "reglink.xls layout changed"
    data = raw.iloc[header_row[0] + 1:, [0, 1]].dropna()
    data.columns = ["fips", "region_code"]
    data = data.astype(int)

    assert len(data) == 3112, "expected 3112 counties, got %d" % len(data)
    assert not data["fips"].duplicated().any(), "duplicate FIPS in crosswalk"
    assert set(data["region_code"]) == set(range(1, 10)), \
        "expected region codes 1..9, got %s" % sorted(set(data["region_code"]))
    data["region_ers"] = data["region_code"].map(ERS_KEY)
    cw = data.set_index("fips")["region_ers"]
    for fips, expected in SPOT_CHECKS:
        got = cw.get(fips)
        assert got == expected, "spot check %d: expected %s, got %s" % (
            fips, expected, got)
    print("crosswalk OK: %d counties, 9 regions, spot checks pass" % len(cw))
    return cw


def read_master(name):
    """Read a master CSV with every column as raw text (values byte-preserved),
    plus an integer join key derived from fips."""
    df = pd.read_csv(REPO_ROOT / name, dtype=str)
    assert not df["fips"].isna().any()
    return df


def replace_region(df, col_name, values):
    """Return df with the `region` column replaced (in place in the column
    order) by `col_name` = values."""
    out = df.copy()
    cols = list(out.columns)
    i = cols.index("region")
    out = out.drop(columns=["region"])
    out.insert(i, col_name, values)
    return out


def main():
    cw = load_crosswalk()
    seasonal = read_master("master_dataset.csv")
    monthly = read_master("master_dataset_monthly.csv")

    key_s = set(zip(seasonal["fips"], seasonal["year"]))
    key_m = set(zip(monthly["fips"], monthly["year"]))
    assert key_s == key_m, "seasonal and monthly datasets disagree on rows"
    assert len(seasonal) == len(monthly)

    # --- Partition 2: county-level ERS regions -----------------------------
    fips_int = seasonal["fips"].astype(int)
    region_ers = fips_int.map(cw)
    n_unmatched = int(region_ers.isna().sum())
    assert n_unmatched == 0, "%d county-years failed to match the crosswalk" % n_unmatched

    cy_counts = region_ers.value_counts()
    print("\ncounty-year count per ERS region (all 8691 rows):")
    print(cy_counts.to_string())
    dropped = cy_counts[cy_counts < MIN_COUNTY_YEARS]
    kept = cy_counts[cy_counts >= MIN_COUNTY_YEARS]
    print("\n*** INCLUSION RULE (>= %d county-years): dropping %d region(s) ***"
          % (MIN_COUNTY_YEARS, len(dropped)))
    for r, n in dropped.items():
        drop_fips = sorted(seasonal.loc[region_ers == r, "fips"].unique())
        print("    DROPPED %-22s %d county-years, counties: %s" % (r, n, drop_fips))
    print("kept %d regions: %s" % (len(kept), list(kept.index)))

    keep_mask = region_ers.isin(kept.index).values
    order = np.arange(len(seasonal))[keep_mask]
    ers_seasonal = replace_region(seasonal.loc[order], "region_ers",
                                  region_ers.loc[order].values)
    # monthly rows are keyed identically but may be ordered differently; align by key
    mkey = pd.MultiIndex.from_frame(monthly[["fips", "year"]])
    skey = pd.MultiIndex.from_frame(seasonal[["fips", "year"]])
    m_region_ers = monthly["fips"].astype(int).map(cw)
    m_keep = m_region_ers.isin(kept.index).values
    ers_monthly = replace_region(monthly.loc[m_keep], "region_ers",
                                 m_region_ers.loc[m_keep].values)
    assert len(ers_seasonal) == len(ers_monthly) == int(keep_mask.sum())
    print("wrote rows: %d (dropped %d of %d)"
          % (len(ers_seasonal), (~keep_mask).sum(), len(seasonal)))

    ers_seasonal.to_csv(REPO_ROOT / "master_dataset_ers.csv", index=False)
    ers_monthly.to_csv(REPO_ROOT / "master_dataset_monthly_ers.csv", index=False)

    # --- Partition 3: climate-derived clusters -----------------------------
    k = len(kept)
    print("\nclimate partition: k-means with k=%d (= surviving ERS regions), "
          "n_init=%d, random_state=%d" % (k, KMEANS_N_INIT, KMEANS_SEED))
    mon_num = pd.read_csv(REPO_ROOT / "master_dataset_monthly.csv")
    feat_cols = [c for c in mon_num.columns
                 if c not in ("fips", "state_name", "county_name", "year",
                              "yield_bu_acre", "state", "region")]
    assert len(feat_cols) == 35, "expected 35 monthly features, got %d" % len(feat_cols)
    normals = mon_num.groupby("fips")[feat_cols].mean()  # per-county, across years
    X = StandardScaler().fit_transform(normals.values)
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=KMEANS_SEED)
    labels = km.fit_predict(X)
    print("k-means inertia: %.1f" % km.inertia_)

    county_cluster = pd.Series(labels, index=normals.index)
    # relabel clusters by descending county-year count for stable, readable names
    cy_per_cluster = mon_num["fips"].map(county_cluster).value_counts()
    rank = {old: i for i, old in enumerate(cy_per_cluster.index)}
    county_name_map = county_cluster.map(lambda c: "Clim-%d" % rank[c])

    clim_s = seasonal["fips"].astype(int).map(county_name_map)
    clim_m = monthly["fips"].astype(int).map(county_name_map)
    assert not clim_s.isna().any() and not clim_m.isna().any()

    tab = pd.DataFrame({
        "counties": county_name_map.value_counts(),
        "county_years": clim_s.value_counts()}).sort_index()
    print("\nclimate cluster sizes:")
    print(tab.to_string())
    small = tab[tab["county_years"] < MIN_COUNTY_YEARS]
    if len(small):
        print("*** WARNING: cluster(s) below %d county-years (kept, noted as "
              "imbalance — not re-run): %s" % (MIN_COUNTY_YEARS, list(small.index)))
    else:
        print("all clusters meet the %d county-year threshold" % MIN_COUNTY_YEARS)

    clim_seasonal = replace_region(seasonal, "region_clim", clim_s.values)
    clim_monthly = replace_region(monthly, "region_clim", clim_m.values)
    clim_seasonal.to_csv(REPO_ROOT / "master_dataset_clim.csv", index=False)
    clim_monthly.to_csv(REPO_ROOT / "master_dataset_monthly_clim.csv", index=False)
    print("wrote master_dataset_clim.csv and master_dataset_monthly_clim.csv "
          "(%d rows each)" % len(clim_seasonal))

    # cross-tab for the write-up: how much do the partitions disagree?
    ct = pd.crosstab(seasonal["region"], region_ers.values)
    print("\ncounty-year cross-tab: state-level region (rows) x ERS region (cols):")
    print(ct.to_string())
    agree = sum(ct.loc[r, r] for r in ct.index if r in ct.columns)
    print("state-level vs ERS agreement: %d/%d county-years (%.1f%%)"
          % (agree, len(seasonal), 100.0 * agree / len(seasonal)))


if __name__ == "__main__":
    main()
