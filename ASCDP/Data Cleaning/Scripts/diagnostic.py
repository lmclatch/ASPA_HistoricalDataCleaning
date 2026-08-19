#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnose the cleaned station files before modelling.

Answers three questions raised by the pipeline dry run:

  1. What non-numeric values are hiding in the numeric columns, and how many
     valid readings are they costing?
  2. Where are the real gaps in each variable, as opposed to the ones
     catalogued in config.GAP_WINDOWS?
  3. When does each station's record actually start and end?

Usage:
    python diagnose.py                 # all stations
    python diagnose.py --station Poloa
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

import config as cfg
from pipeline import find_station_file

NUMERIC_COLUMNS = ["PTemp_C_Max", "AirTF_Avg", "SlrW_Avg", "SlrMJ_Tot",
                   "RH", "Rain_in_Tot", "WS_mph_S_WVT", "WindDir_D1_WVT",
                   "WindDir_SD1_WVT"]

MIN_GAP_ROWS = 96          # 96 x 15min = 1 day; shorter runs are sensor blips
TOP_GAPS_SHOWN = 8


def non_numeric_values(series: pd.Series) -> Counter:
    """Values that survive as strings but fail numeric conversion."""
    if series.dtype.kind in "fiu":
        return Counter()
    coerced = pd.to_numeric(series, errors="coerce")
    bad = series[coerced.isna() & series.notna()]
    return Counter(bad.astype(str).str.strip())


def find_gaps(ts: pd.Series, valid: pd.Series) -> pd.DataFrame:
    """Contiguous runs where the variable is missing, longest first."""
    missing = ~valid
    if not missing.any():
        return pd.DataFrame(columns=["start", "end", "rows", "days"])

    # Label each contiguous run of equal values.
    run_id = (missing != missing.shift()).cumsum()
    runs = []
    for _, idx in missing[missing].groupby(run_id[missing]):
        block = ts[idx.index]
        rows = len(block)
        if rows < MIN_GAP_ROWS:
            continue
        runs.append({
            "start": block.min(),
            "end": block.max(),
            "rows": rows,
            "days": round(rows * 15 / 60 / 24, 1),
        })

    if not runs:
        return pd.DataFrame(columns=["start", "end", "rows", "days"])
    return pd.DataFrame(runs).sort_values("rows", ascending=False)


def configured_windows(station: str, variable: str) -> list:
    key = (station, variable)
    if key not in cfg.GAP_WINDOWS:
        return []
    w = cfg.GAP_WINDOWS[key]
    return [(pd.to_datetime(w["start"]),
             pd.to_datetime(w["end"]) if w["end"] else None)]


def overlaps_configured(gap_start, gap_end, windows) -> bool:
    for w_start, w_end in windows:
        if w_end is None:
            if gap_end >= w_start:
                return True
        elif gap_start <= w_end and gap_end >= w_start:
            return True
    return False


def diagnose_station(station: str) -> None:
    path = find_station_file(station)
    if path is None:
        print(f"  no file found for {station}")
        return

    raw = pd.read_csv(path, low_memory=False)
    raw["TIMESTAMP"] = pd.to_datetime(raw["TIMESTAMP"], errors="coerce")
    raw = raw[raw["TIMESTAMP"].notna()].sort_values("TIMESTAMP")

    print(f"\n{'=' * 74}")
    print(f"{station}   {path.name}   {len(raw):,} rows")
    print(f"record: {raw['TIMESTAMP'].min()}  ->  {raw['TIMESTAMP'].max()}")

    expected = pd.date_range(raw["TIMESTAMP"].min(), raw["TIMESTAMP"].max(),
                             freq="15min")
    print(f"expected rows at 15min: {len(expected):,}  "
          f"(missing timestamps: {len(expected) - len(raw):,})")

    # --- 1. Non-numeric contamination -------------------------------------
    print("\n  non-numeric values in numeric columns")
    found_any = False
    for col in NUMERIC_COLUMNS:
        if col not in raw.columns:
            continue
        bad = non_numeric_values(raw[col])
        if bad:
            found_any = True
            total = sum(bad.values())
            shown = ", ".join(f"{v!r} x{n:,}" for v, n in bad.most_common(4))
            print(f"    {col:<18} {total:>8,} rows  ({shown})")
    if not found_any:
        print("    none -- all columns parse cleanly as numeric")

    # --- 2. Real gaps vs configured gaps ----------------------------------
    print(f"\n  missing-data runs longer than "
          f"{MIN_GAP_ROWS * 15 / 60 / 24:.0f} day(s)")
    for col in NUMERIC_COLUMNS:
        if col not in raw.columns:
            continue
        valid = pd.to_numeric(raw[col], errors="coerce").notna()
        pct = (~valid).mean() * 100
        gaps = find_gaps(raw["TIMESTAMP"].reset_index(drop=True),
                         valid.reset_index(drop=True))
        windows = configured_windows(station, col)

        header = f"    {col:<18} {pct:5.1f}% missing overall"
        if gaps.empty:
            print(header + "  -- no long runs")
            continue
        print(header + f"  -- {len(gaps)} long run(s)")

        for _, g in gaps.head(TOP_GAPS_SHOWN).iterrows():
            known = overlaps_configured(g["start"], g["end"], windows)
            tag = "configured" if known else "NOT IN CONFIG"
            print(f"        {g['start']}  ->  {g['end']}  "
                  f"{g['days']:>7.1f}d  [{tag}]")
        if len(gaps) > TOP_GAPS_SHOWN:
            other = gaps.iloc[TOP_GAPS_SHOWN:]["rows"].sum()
            print(f"        ... {len(gaps) - TOP_GAPS_SHOWN} shorter run(s), "
                  f"{other:,} rows total")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", help="One station; default is all.")
    args = p.parse_args()

    stations = [args.station] if args.station else cfg.STATIONS
    for s in stations:
        diagnose_station(s)

    print(f"\n{'=' * 74}")
    print("Runs marked NOT IN CONFIG are real outages missing from "
          "config.GAP_WINDOWS.\nEach either needs a config entry, or explains "
          "why a target looks sparse.")


if __name__ == "__main__":
    main()