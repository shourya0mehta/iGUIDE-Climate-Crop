"""Download WRF-HRRR computed weather from CropNet and aggregate to county level.

Source: https://huggingface.co/datasets/CropNet/CropNet (public, no auth).
Fetched with plain requests.get — the cropnet package, hf_hub_download, and the
datasets library all fail on this repo; spaces in paths must stay %20-encoded.

For each (year, state): fetch the 12 monthly CSVs (a 404 means the month is
absent upstream and is skipped), keep Daily rows in the April-October growing
season, and aggregate per county FIPS. AL 2017 is entirely absent upstream and
is the only expected full-state-year SKIP.

Resumable: after every completed state-year the running result is rewritten to
hrrr_checkpoint.csv; on startup any (state, year) already checkpointed is
skipped, so rerunning resumes instead of restarting.

Output: hrrr_growing_season_2017_2022.csv (18304 rows x 8 cols).
"""
import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

BASE = "https://huggingface.co/datasets/CropNet/CropNet"
DATA_DIR = Path(__file__).resolve().parent
CHECKPOINT = DATA_DIR / "hrrr_checkpoint.csv"
OUTPUT = DATA_DIR / "hrrr_growing_season_2017_2022.csv"

YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

STATE_FIPS = {
    'AL': '01', 'AR': '05', 'AZ': '04', 'CA': '06', 'CO': '08', 'CT': '09', 'DE': '10',
    'FL': '12', 'GA': '13', 'IA': '19', 'ID': '16', 'IL': '17', 'IN': '18', 'KS': '20',
    'KY': '21', 'LA': '22', 'MA': '25', 'MD': '24', 'ME': '23', 'MI': '26', 'MN': '27',
    'MO': '29', 'MS': '28', 'MT': '30', 'NC': '37', 'ND': '38', 'NE': '31', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NV': '32', 'NY': '36', 'OH': '39', 'OK': '40', 'OR': '41',
    'PA': '42', 'RI': '44', 'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VA': '51', 'VT': '50', 'WA': '53', 'WI': '55', 'WV': '54', 'WY': '56',
}

COLUMNS = ['FIPS Code', 'avg_temp', 'total_precip', 'avg_humidity',
           'avg_radiation', 'avg_vpd', 'year', 'state']
EXPECTED_SHAPE = (18304, 8)

_local = threading.local()


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def fetch_month(year, state, month):
    """One monthly CSV as a DataFrame, or None if absent upstream (404)."""
    url = (f"{BASE}/resolve/main/WRF-HRRR%20Computed%20Dataset/data/"
           f"{year}/{state}/HRRR_{STATE_FIPS[state]}_{state}_{year}-{month:02d}.csv")
    last_err = None
    for attempt in range(4):
        try:
            r = _session().get(url, timeout=180)
        except requests.RequestException as e:
            last_err = str(e)
        else:
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                return pd.read_csv(io.StringIO(r.text))
            last_err = f"HTTP {r.status_code}"
        time.sleep(2 * 2 ** attempt)
    # A transient failure must not masquerade as a 404: raise so the run dies
    # loudly and the checkpoint resumes it, rather than silently dropping
    # months and corrupting the state-year aggregate.
    raise RuntimeError(f"{url}: still failing after retries ({last_err})")


def fetch_state_year(year, state):
    """Aggregate one state-year to county level; None if no months exist."""
    with ThreadPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(lambda m: fetch_month(year, state, m), range(1, 13)))
    frames = [f for f in frames if f is not None]
    if not frames:
        return None, 0
    combined = pd.concat(frames, ignore_index=True)
    daily = combined[combined['Daily/Monthly'] == 'Daily']
    growing = daily[daily['Month'].between(4, 10)]  # April-October, inclusive
    agg = growing.groupby('FIPS Code').agg(
        avg_temp=('Avg Temperature (K)', 'mean'),
        total_precip=('Precipitation (kg m**-2)', 'sum'),  # sum — the rest are means
        avg_humidity=('Relative Humidity (%)', 'mean'),
        avg_radiation=('Downward Shortwave Radiation Flux (W m**-2)', 'mean'),
        avg_vpd=('Vapor Pressure Deficit (kPa)', 'mean'),
    ).reset_index()
    agg['year'] = year
    agg['state'] = state
    return agg, len(frames)


def main():
    frames = []
    done = set()
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        frames.append(ckpt)
        done = set(zip(ckpt['state'], ckpt['year']))
        print(f"Resuming: {len(done)} state-years already in checkpoint", flush=True)

    states = sorted(STATE_FIPS)
    total = len(YEARS) * len(states)
    n_done = len(done)
    t0 = time.time()
    for year in YEARS:
        for state in states:
            if (state, year) in done:
                continue
            agg, n_months = fetch_state_year(year, state)
            n_done += 1
            if agg is None:
                print(f"SKIP {state} {year}: no monthly files upstream", flush=True)
                continue
            frames.append(agg)
            pd.concat(frames, ignore_index=True)[COLUMNS].to_csv(CHECKPOINT, index=False)
            print(f"[{n_done}/{total}] {state} {year}: {len(agg)} counties from "
                  f"{n_months} months ({time.time() - t0:.0f}s elapsed)", flush=True)

    result = pd.concat(frames, ignore_index=True)[COLUMNS]
    assert result.shape == EXPECTED_SHAPE, (
        f"expected {EXPECTED_SHAPE}, got {result.shape} — upstream data has "
        f"drifted; output NOT written (checkpoint retained)")
    result.to_csv(OUTPUT, index=False)
    print(f"OK: wrote {OUTPUT} ({result.shape[0]} rows, {result.shape[1]} cols)",
          flush=True)


if __name__ == "__main__":
    main()
