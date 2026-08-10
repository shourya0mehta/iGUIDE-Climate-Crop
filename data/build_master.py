"""Merge USDA yields with HRRR growing-season weather into master_dataset.csv.

Regions are USDA ERS Farm Resource Regions approximated at the state level —
a known approximation (real regions are county-level and cut across state
lines); the county-level mapping is tracked as separate work.

Every acceptance check runs before anything is written: on any mismatch the
script exits WITHOUT writing master_dataset.csv, because a drifted dataset
would invalidate comparison against results already recorded.
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
USDA_CSV = DATA_DIR / "usda_corn_yields_2017_2022.csv"
HRRR_CSV = DATA_DIR / "hrrr_growing_season_2017_2022.csv"
OUTPUT = DATA_DIR.parent / "master_dataset.csv"

REGION_MAP = {
    # Northern Crescent
    'ME': 'Northern Crescent', 'NH': 'Northern Crescent', 'VT': 'Northern Crescent',
    'MA': 'Northern Crescent', 'RI': 'Northern Crescent', 'CT': 'Northern Crescent',
    'NY': 'Northern Crescent', 'NJ': 'Northern Crescent', 'PA': 'Northern Crescent',
    'OH': 'Northern Crescent', 'IN': 'Northern Crescent', 'MI': 'Northern Crescent',
    'WI': 'Northern Crescent', 'MN': 'Northern Crescent',
    # Northern Great Plains
    'ND': 'Northern Great Plains', 'SD': 'Northern Great Plains',
    'NE': 'Northern Great Plains', 'KS': 'Northern Great Plains',
    'MT': 'Northern Great Plains', 'WY': 'Northern Great Plains',
    'CO': 'Northern Great Plains',
    # Heartland
    'MO': 'Heartland', 'IL': 'Heartland', 'IA': 'Heartland',
    # Prairie Gateway
    'TX': 'Prairie Gateway', 'OK': 'Prairie Gateway',
    # Southern Seaboard
    'AR': 'Southern Seaboard', 'LA': 'Southern Seaboard', 'MS': 'Southern Seaboard',
    'AL': 'Southern Seaboard', 'GA': 'Southern Seaboard', 'SC': 'Southern Seaboard',
    'NC': 'Southern Seaboard', 'VA': 'Southern Seaboard', 'TN': 'Southern Seaboard',
    'KY': 'Southern Seaboard', 'WV': 'Southern Seaboard', 'MD': 'Southern Seaboard',
    'DE': 'Southern Seaboard', 'FL': 'Southern Seaboard',
    # Basin and Range
    'CA': 'Basin and Range', 'NV': 'Basin and Range', 'AZ': 'Basin and Range',
    'NM': 'Basin and Range', 'UT': 'Basin and Range', 'ID': 'Basin and Range',
    'OR': 'Basin and Range', 'WA': 'Basin and Range',
}

COLUMNS = ['fips', 'state_name', 'county_name', 'year', 'yield_bu_acre',
           'avg_temp', 'total_precip', 'avg_humidity', 'avg_radiation',
           'avg_vpd', 'state', 'region']

EXPECTED_REGION_COUNTS = {
    'Southern Seaboard': 2681,
    'Northern Crescent': 2554,
    'Heartland': 1477,
    'Northern Great Plains': 1426,
    'Prairie Gateway': 454,
    'Basin and Range': 99,
}


def main():
    usda = pd.read_csv(USDA_CSV)
    hrrr = pd.read_csv(HRRR_CSV)

    hrrr = hrrr.rename(columns={'FIPS Code': 'fips'})
    hrrr['fips'] = hrrr['fips'].astype(str).str.zfill(5)
    usda['fips'] = usda['fips'].astype(str).str.zfill(5)
    merged = pd.merge(usda, hrrr, on=['fips', 'year'], how='inner')
    merged['region'] = merged['state'].map(REGION_MAP)

    errors = []
    if list(merged.columns) != COLUMNS:
        errors.append(f"column order is {list(merged.columns)}")
    if merged.shape != (8691, 12):
        errors.append(f"shape is {merged.shape}, expected (8691, 12)")
    nulls = merged.isnull().sum()
    if nulls.any():
        errors.append("nulls present: %s" % nulls[nulls > 0].to_dict())
    if merged['region'].isnull().any():
        unmapped = sorted(merged.loc[merged['region'].isnull(), 'state'].unique())
        errors.append(f"unmapped region for states: {unmapped}")
    counts = merged['region'].value_counts().to_dict()
    if counts != EXPECTED_REGION_COUNTS:
        errors.append(f"region counts {counts}, expected {EXPECTED_REGION_COUNTS}")
    spot = merged[(merged['fips'] == '17019') & (merged['year'] == 2019)]
    if len(spot) != 1 or spot['yield_bu_acre'].iloc[0] != 180.8:
        errors.append("Champaign IL 2019 spot check failed: %s"
                      % spot.to_dict('records'))

    if errors:
        print("MISMATCH — master_dataset.csv NOT written; upstream data or the "
              "spec has drifted:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    merged.to_csv(OUTPUT, index=False)
    print(f"OK: wrote {OUTPUT} ({merged.shape[0]} rows, {merged.shape[1]} cols)")
    print("Region counts:")
    print(merged['region'].value_counts().to_string())


if __name__ == "__main__":
    main()
