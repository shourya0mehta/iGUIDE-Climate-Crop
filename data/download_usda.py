"""Download USDA county-level corn yields (2017-2022) from CropNet.

Source: https://huggingface.co/datasets/CropNet/CropNet (public, no auth).
Fetched with plain requests.get — the cropnet package, hf_hub_download, and the
datasets library all fail on this repo; spaces in paths must stay %20-encoded.

Output: usda_corn_yields_2017_2022.csv (8739 rows x 5 cols).
"""
import io
from pathlib import Path

import pandas as pd
import requests

BASE = "https://huggingface.co/datasets/CropNet/CropNet"
DATA_DIR = Path(__file__).resolve().parent
OUTPUT = DATA_DIR / "usda_corn_yields_2017_2022.csv"

# Known-correct per-year row counts; a mismatch means upstream data drifted.
EXPECTED_ROWS = {2017: 1481, 2018: 1345, 2019: 1257, 2020: 1668, 2021: 1472,
                 2022: 1516}
EXPECTED_TOTAL = 8739
EXPECTED_UNIQUE_FIPS = 1903


def fetch_year(year):
    url = (f"{BASE}/resolve/main/USDA%20Crop%20Dataset/Corn/{year}/"
           f"USDA_Corn_County_{year}.csv")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    assert len(df) == EXPECTED_ROWS[year], (
        f"{year}: expected {EXPECTED_ROWS[year]} rows, got {len(df)} — "
        f"upstream data has drifted")
    print(f"{year}: {len(df)} rows", flush=True)
    return df


def main():
    df = pd.concat([fetch_year(y) for y in sorted(EXPECTED_ROWS)],
                   ignore_index=True)
    assert len(df) == EXPECTED_TOTAL, f"expected {EXPECTED_TOTAL}, got {len(df)}"

    df['fips'] = (df['state_ansi'].astype(str).str.zfill(2)
                  + df['county_ansi'].astype(str).str.zfill(3))
    df = df.rename(columns={'YIELD, MEASURED IN BU / ACRE': 'yield_bu_acre'})
    df = df[['fips', 'state_name', 'county_name', 'year', 'yield_bu_acre']]

    assert df.shape == (EXPECTED_TOTAL, 5), f"bad shape {df.shape}"
    n_fips = df['fips'].nunique()
    assert n_fips == EXPECTED_UNIQUE_FIPS, (
        f"expected {EXPECTED_UNIQUE_FIPS} unique FIPS, got {n_fips}")

    df.to_csv(OUTPUT, index=False)
    print(f"OK: wrote {OUTPUT} ({df.shape[0]} rows, {df.shape[1]} cols, "
          f"{n_fips} unique FIPS)", flush=True)


if __name__ == "__main__":
    main()
