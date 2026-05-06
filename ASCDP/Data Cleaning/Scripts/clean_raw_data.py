"""
Clean raw weather station data from the Hydro_Monitoring_Network_ASPA-UH repo.

Pipeline:
    1. Fetch ALL_15min_data.csv and Bad_data.csv files for each station from GitHub.
    2. Mask out known bad data ranges (set to NaN).
    3. Convert imperial units to metric in-place (column names preserved -- see README).
    4. Save one cleaned CSV per station.

NOTE ON COLUMN NAMES:
    Three columns retain their original imperial-suffixed names but contain
    metric values after this script runs:
        - AirTF_Avg     -> Celsius (was Fahrenheit)
        - Rain_in_Tot   -> millimeters (was inches)
        - WS_mph_S_WVT  -> meters/second (was mph)
    See README.md for rationale.
"""

from __future__ import annotations

import argparse
import logging
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------------------- 
# Configuration
# -----------------------------------------------------------------------------

GITHUB_API_URL = (
    "https://api.github.com/repos/cshuler/Hydro_Monitoring_Network_ASPA-UH/"
    "contents/Scripts/Liza_Aimee_Workspace/Data"
)

# Columns that need imperial -> metric conversion.
# (column_name, conversion_function)
UNIT_CONVERSIONS = {
    "AirTF_Avg":    lambda f:   (f - 32.0) * 5.0 / 9.0,   # F -> C
    "Rain_in_Tot":  lambda inches: inches * 25.4,         # in -> mm
    "WS_mph_S_WVT": lambda mph: mph * 0.44704,            # mph -> m/s
}

# Default output directory: <repo>/Data Cleaning/Cleaned Raw Data/
# (resolved relative to this script's location)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent / "Cleaned Raw Data"
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- 
# Fetching
# -----------------------------------------------------------------------------

def list_csv_files_on_github(api_url: str) -> dict[str, str]:
    """Return {filename: download_url} for every .csv file in a GitHub directory."""
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    files = response.json()
    return {
        f["name"]: f["download_url"]
        for f in files
        if f["name"].endswith(".csv") and "download_url" in f
    }


def fetch_all_csv(url: str) -> pd.DataFrame:
    """
    Fetch an ALL_15min_data CSV.

    These files have a 3-row header:
        row 0: column names      <- keep as header
        row 1: units (Volts, Deg C, etc.)
        row 2: aggregation type (Avg, Max, Tot, etc.)
        row 3+: actual data
    """
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), skiprows=[1, 2])


def fetch_bad_data_csv(url: str) -> pd.DataFrame:
    """Fetch a Bad_data CSV. Single-row header, no skipping needed."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text))


# ----------------------------------------------------------------------------- 
# Station name parsing & file pairing
# -----------------------------------------------------------------------------

def station_name_from_filename(filename: str) -> str | None:
    """
    Extract the station name from a filename.

    Examples:
        'Aasu_ALL_15min_data.csv' -> 'Aasu'
        'Aasu_Bad data.csv'        -> 'Aasu'
    """
    match = re.match(r"([A-Za-z0-9]+)", filename)
    return match.group(1) if match else None


def pair_files_by_station(
    csv_files: dict[str, str],
) -> dict[str, dict[str, str]]:
    """
    Group CSV download URLs by station name.

    Returns:
        {station: {"all": url, "bad": url}}
    Stations missing either file are logged and excluded.
    """
    paired: dict[str, dict[str, str]] = {}
    for filename, url in csv_files.items():
        station = station_name_from_filename(filename)
        if station is None:
            logger.warning("Could not parse station name from %s, skipping.", filename)
            continue
        entry = paired.setdefault(station, {})
        upper = filename.upper()
        if "ALL" in upper:
            entry["all"] = url
        elif "BAD" in upper:
            entry["bad"] = url

    complete = {s: e for s, e in paired.items() if "all" in e and "bad" in e}
    incomplete = set(paired) - set(complete)
    for s in incomplete:
        logger.warning("Station %s is missing an ALL or Bad file; skipping.", s)
    return complete


# ----------------------------------------------------------------------------- 
# Bad-data masking
# -----------------------------------------------------------------------------

def apply_bad_data_mask(
    all_df: pd.DataFrame,
    bad_df: pd.DataFrame,
    station: str,
) -> pd.DataFrame:
    """
    Set values to NaN for date ranges flagged as bad.

    `Data affected` may be:
        - 'ALL' or 'ALL data' (any string starting with 'ALL'): mask every column
           except TIMESTAMP, LAT, LON, RECORD.
        - A specific column name: mask just that column.
    """
    if "TIMESTAMP" not in all_df.columns:
        logger.error("TIMESTAMP column missing for %s; cannot apply bad-data mask.", station)
        return all_df

    all_df = all_df.copy()
    all_df["TIMESTAMP"] = pd.to_datetime(all_df["TIMESTAMP"], errors="coerce")
    bad_df = bad_df.copy()
    bad_df["Bad data Start"] = pd.to_datetime(bad_df["Bad data Start"], errors="coerce")
    bad_df["Bad data End"]   = pd.to_datetime(bad_df["Bad data End"],   errors="coerce")

    # Columns that should NEVER be masked even on an "ALL" row.
    metadata_cols = {"TIMESTAMP", "LAT", "LON", "RECORD"}

    masked_count = 0
    for _, row in bad_df.iterrows():
        affected = str(row.get("Data affected", "")).strip()
        start, end = row["Bad data Start"], row["Bad data End"]

        if pd.isnull(start) or pd.isnull(end):
            continue

        time_mask = (all_df["TIMESTAMP"] >= start) & (all_df["TIMESTAMP"] <= end)

        if affected.upper().startswith("ALL"):
            cols_to_mask = [c for c in all_df.columns if c not in metadata_cols]
            all_df.loc[time_mask, cols_to_mask] = np.nan
            logger.info(
                "[%s] Masked ALL columns from %s to %s (%d rows)",
                station, start, end, time_mask.sum(),
            )
        elif affected in all_df.columns:
            all_df.loc[time_mask, affected] = np.nan
            logger.info(
                "[%s] Masked %s from %s to %s (%d rows)",
                station, affected, start, end, time_mask.sum(),
            )
        else:
            logger.warning(
                "[%s] 'Data affected' value %r not found in columns; skipping row.",
                station, affected,
            )
            continue
        masked_count += 1

    logger.info("[%s] Applied %d bad-data mask rules.", station, masked_count)
    return all_df


# ----------------------------------------------------------------------------- 
# Unit conversion
# -----------------------------------------------------------------------------

def convert_to_metric(df: pd.DataFrame, station: str) -> pd.DataFrame:
    """
    Convert imperial-unit columns to metric in place.

    Column names are preserved (e.g. Rain_in_Tot still says 'in_' but holds mm).
    NaN values pass through unchanged.
    """
    df = df.copy()
    for col, convert in UNIT_CONVERSIONS.items():
        if col not in df.columns:
            logger.warning("[%s] Expected column %s not found; skipping conversion.", station, col)
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = convert(df[col])
    logger.info("[%s] Unit conversion complete.", station)
    return df


# ----------------------------------------------------------------------------- 
# Orchestration
# -----------------------------------------------------------------------------

def clean_station(station: str, all_url: str, bad_url: str, output_dir: Path) -> Path:
    """Run the full clean for one station and write the result. Returns the output path."""
    logger.info("[%s] Fetching ALL data...", station)
    all_df = fetch_all_csv(all_url)
    logger.info("[%s] Fetching Bad data...", station)
    bad_df = fetch_bad_data_csv(bad_url)

    all_df = apply_bad_data_mask(all_df, bad_df, station)
    all_df = convert_to_metric(all_df, station)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{station}_cleaned.csv"
    all_df.to_csv(output_path, index=False)
    logger.info("[%s] Saved %d rows to %s", station, len(all_df), output_path)
    return output_path


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    csv_files = list_csv_files_on_github(GITHUB_API_URL)
    if not csv_files:
        raise RuntimeError("No CSV files found at the GitHub API URL.")
    logger.info("Found %d CSV files in remote repo.", len(csv_files))

    paired = pair_files_by_station(csv_files)
    logger.info("Paired %d stations: %s", len(paired), ", ".join(sorted(paired)))

    for station, urls in sorted(paired.items()):
        try:
            clean_station(station, urls["all"], urls["bad"], output_dir)
        except Exception as e:
            logger.exception("[%s] FAILED: %s", station, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write cleaned CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)