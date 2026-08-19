#!/usr/bin/env python3
"""
Download Synoptic Data mesonet observations for the ASPA gap-filling pipeline.

Scope: this script's ONLY job is to fetch synoptic station data at its native
resolution (~hourly) and write it to disk faithfully. It does NOT resample to
15-min, join to ASPA data, or build features -- that belongs in the feature
engineering step downstream.

Outputs, per station, into OUT_DIR:
    <STID>_<START_YEAR>-<END_YEAR>.csv      cleaned, metric, hourly precip derived
    _raw_json/<STID>_<YEAR>.json            cached API response (re-run safe)

Usage:
    python download_synoptic.py
    python download_synoptic.py --stations NSTU --start-year 2023 --end-year 2024
    python download_synoptic.py --force          # ignore cache, re-hit the API

Requires a Synoptic API token in a .env file at the repo root:
    SYNOPTIC_TOKEN=your_token_here
(.env must be gitignored.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

STATIONS = ["NSTU", "SFGP6"]

START_YEAR = 2017
END_YEAR = 2024

# Requested from the API. Stations that don't report a variable simply omit it
# from the response -- we keep whatever comes back and log what's missing.
VARIABLES = [
    "air_temp",
    "relative_humidity",
    "wind_speed",
    "wind_direction",
    "precip_accum_one_hour",   # preferred precip source
    "precip_accum",            # fallback, diffed in post-processing
    "solar_radiation",
    "pressure",
]

# American Samoa is UTC-11 year round, no DST, so "local" is unambiguous.
OBTIMEZONE = "local"

# Write TIMESTAMP as naive local time (drops the -1100 offset). Set False to
# keep the offset in the ISO string, matching a manual Synoptic CSV download.
# Naive is easier to join against the ASPA 15-min timestamps downstream.
STRIP_TZ = True

# Synoptic returns ELEVATION in FEET regardless of the units parameter.
# The ASPA pipeline is all-metric, so convert.
ELEVATION_TO_METERS = True

BASE_URL = "https://api.synopticdata.com/v2/stations/timeseries"

# This script lives at <repo>/ASCDP/Data Cleaning/Scripts/, so relative to
# SCRIPT_DIR (= Scripts):
#   parents[0] = Data Cleaning
#   parents[1] = ASCDP
#   parents[2] = repo root (where .env lives)
SCRIPT_DIR = Path(__file__).resolve().parent
ASCDP_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
OUT_DIR = ASCDP_DIR / "Raw Input Data" / "Synoptic"

MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5
REQUEST_TIMEOUT_SEC = 120


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def get_token() -> str:
    load_dotenv(REPO_ROOT / ".env")
    token = os.getenv("SYNOPTIC_TOKEN")
    if not token:
        sys.exit(
            "No SYNOPTIC_TOKEN found.\n"
            f"Create {REPO_ROOT / '.env'} containing:\n"
            "    SYNOPTIC_TOKEN=your_token_here\n"
            "and make sure .env is in .gitignore."
        )
    return token


def _redact(text: str, token: str) -> str:
    """Keep the token out of anything we print. Tracebacks and log files
    outlive the terminal session they were created in."""
    return text.replace(token, "***REDACTED***") if token else text


def fetch_station_year(stid: str, year: int, token: str) -> dict:
    """One API call: a single station, a single calendar year.

    Chunking by year keeps each response small enough to be reliable over an
    8-year range, and makes the on-disk cache granular enough that a failure
    part-way through doesn't cost you the whole download.

    start/end are interpreted as UTC by the API. We pad by a day on each side
    and trim after parsing, so no local-time observations fall off the edges.
    """
    params = {
        "stid": stid,
        "start": f"{year - 1}1231{'0000'}",
        "end": f"{year + 1}0101{'2359'}",
        "vars": ",".join(VARIABLES),
        "obtimezone": OBTIMEZONE,
        "token": token,
        # units default to metric -- do not pass units=english
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)

            # 4xx means the request itself is wrong (bad token, tier limit,
            # unavailable station). Retrying cannot fix it, so fail fast with
            # a readable message instead of hammering the API three more times.
            if 400 <= resp.status_code < 500:
                raise PermissionError(
                    f"HTTP {resp.status_code} for {stid} {year}. "
                    "Common causes: token not valid or not yet provisioned; "
                    "requested date range outside your access tier; station "
                    "not included in your plan. Try a small recent request "
                    "first:  .../v2/stations/latest?stid=" + stid
                )

            resp.raise_for_status()
            payload = resp.json()
        except PermissionError:
            raise
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Request failed for {stid} {year}: "
                    f"{_redact(str(exc), token)}"
                ) from None
            wait = RETRY_BACKOFF_SEC * attempt
            print(f"    attempt {attempt} failed "
                  f"({_redact(str(exc), token)}); retrying in {wait}s")
            time.sleep(wait)
            continue

        code = str(payload.get("SUMMARY", {}).get("RESPONSE_CODE", ""))
        if code != "1":
            msg = payload.get("SUMMARY", {}).get("RESPONSE_MESSAGE", "unknown error")
            # Code 2 = no data for this station/period. Not fatal; some stations
            # simply weren't reporting in the earlier years of the range.
            if "no data" in msg.lower() or code == "2":
                print(f"    no data returned for {stid} {year} ({msg})")
                return {}
            raise RuntimeError(f"Synoptic API error for {stid} {year}: {msg}")

        return payload

    return {}


# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def payload_to_frame(payload: dict) -> pd.DataFrame:
    """Flatten one Synoptic JSON response into a tidy DataFrame."""
    stations = payload.get("STATION") or []
    frames = []

    for station in stations:
        obs = station.get("OBSERVATIONS") or {}
        if "date_time" not in obs:
            continue

        df = pd.DataFrame(obs)
        df = df.rename(columns={"date_time": "TIMESTAMP"})
        df["Station_ID"] = station.get("STID")
        df["LAT"] = pd.to_numeric(station.get("LATITUDE"), errors="coerce")
        df["LON"] = pd.to_numeric(station.get("LONGITUDE"), errors="coerce")

        elev = pd.to_numeric(station.get("ELEVATION"), errors="coerce")
        if ELEVATION_TO_METERS and pd.notna(elev):
            elev = round(elev * 0.3048, 2)
        df["Elevation"] = elev

        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def derive_hourly_precip(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a single `precip_hourly_mm` column regardless of what the
    station reports.

    Preference order:
      1. precip_accum_one_hour_set_1 -- already a per-hour total, use as-is.
      2. precip_accum_set_1          -- running accumulation; diff consecutive
                                        observations. Accumulators reset (often
                                        at local midnight or on power cycle),
                                        which shows up as a negative diff; those
                                        are clamped to 0 rather than treated as
                                        negative rainfall.

    Rows where neither is available get NaN, not 0 -- "we don't know" and
    "it didn't rain" must stay distinguishable for the trees downstream.
    """
    if df.empty:
        return df

    hourly_col = next(
        (c for c in df.columns if c.startswith("precip_accum_one_hour")), None
    )
    accum_col = next(
        (c for c in df.columns
         if c.startswith("precip_accum") and not c.startswith("precip_accum_one_hour")),
        None,
    )

    if hourly_col:
        df["precip_hourly_mm"] = pd.to_numeric(df[hourly_col], errors="coerce")
        df["precip_source"] = "one_hour"
    elif accum_col:
        accum = pd.to_numeric(df[accum_col], errors="coerce")
        diffed = accum.groupby(df["Station_ID"]).diff()
        df["precip_hourly_mm"] = diffed.clip(lower=0)
        df["precip_source"] = "diffed_accum"
    else:
        df["precip_hourly_mm"] = pd.NA
        df["precip_source"] = "unavailable"

    return df


def tidy_frame(df: pd.DataFrame, year_lo: int, year_hi: int) -> pd.DataFrame:
    """Sort, deduplicate, trim the padding, and order the columns."""
    if df.empty:
        return df

    ts = pd.to_datetime(df["TIMESTAMP"], format="ISO8601", utc=True)
    local = ts.dt.tz_convert("Pacific/Pago_Pago")
    df["TIMESTAMP"] = local.dt.tz_localize(None) if STRIP_TZ else local

    # Trim the one-day pad we added on each side of every yearly request.
    in_range = (local.dt.year >= year_lo) & (local.dt.year <= year_hi)
    df = df[in_range]

    df = (
        df.sort_values(["Station_ID", "TIMESTAMP"])
          .drop_duplicates(subset=["Station_ID", "TIMESTAMP"], keep="first")
          .reset_index(drop=True)
    )

    lead = ["Station_ID", "TIMESTAMP"]
    tail = ["LAT", "LON", "Elevation"]
    middle = [c for c in df.columns if c not in lead + tail]
    return df[lead + middle + tail]


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def download_station(stid: str, years: range, token: str, force: bool,
                     out_dir: Path) -> pd.DataFrame:
    cache_dir = out_dir / "_raw_json"
    cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for year in years:
        cache_path = cache_dir / f"{stid}_{year}.json"

        if cache_path.exists() and not force:
            print(f"  {year}: cached")
            payload = json.loads(cache_path.read_text())
        else:
            print(f"  {year}: requesting")
            payload = fetch_station_year(stid, year, token)
            cache_path.write_text(json.dumps(payload))
            time.sleep(1)  # be polite to the API

        frame = payload_to_frame(payload)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = derive_hourly_precip(df)
    return tidy_frame(df, years.start, years.stop - 1)


def report_coverage(stid: str, df: pd.DataFrame) -> None:
    """Print what actually came back, so surprises surface here rather than
    three scripts downstream."""
    print(f"\n  {stid}: {len(df):,} observations")
    if df.empty:
        return
    print(f"  range: {df['TIMESTAMP'].min()} -> {df['TIMESTAMP'].max()}")
    print(f"  precip source: {df['precip_source'].iloc[0]}")

    requested = set(VARIABLES)
    returned = {c.rsplit("_set_", 1)[0] for c in df.columns if "_set_" in c}
    missing = sorted(requested - returned)
    if missing:
        print(f"  NOT reported by this station: {', '.join(missing)}")

    for col in sorted(c for c in df.columns if "_set_" in c):
        pct = df[col].isna().mean() * 100
        print(f"    {col:<40} {pct:5.1f}% missing")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stations", nargs="+", default=STATIONS)
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached JSON and re-request from the API.")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    token = get_token()
    years = range(args.start_year, args.end_year + 1)

    for stid in args.stations:
        print(f"\n{stid} ({args.start_year}-{args.end_year})")
        df = download_station(stid, years, token, args.force, out_dir)

        if df.empty:
            print(f"  {stid}: nothing returned, skipping write")
            continue

        out_path = out_dir / f"{stid}_{args.start_year}-{args.end_year}.csv"
        df.to_csv(out_path, index=False)
        report_coverage(stid, df)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()