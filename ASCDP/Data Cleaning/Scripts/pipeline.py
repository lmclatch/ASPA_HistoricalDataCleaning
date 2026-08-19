#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data assimilation pipeline for the ASPA gap-filling project.

Replaces the per-station, comment-and-uncomment assimilation scripts. Reads
every run parameter from config.py; contains no station names, variable names,
dates, or absolute paths of its own.

Usage:
    python pipeline.py --station Aasu --variable Rain_in_Tot
    python pipeline.py --all                 # every configured met run
    python pipeline.py --all --dry-run       # report shapes, write nothing

Outputs, into config.MODEL_INPUT_DIR:
    <station>_<variable>_train.csv
    <station>_<variable>_pred.csv

Changes from the original script, all of which affect results:

  1. Synoptic timestamps are CONVERTED to local Samoa time, not relabelled as
     UTC. The original stripped the -1100 offset after converting to UTC,
     shifting every synoptic feature 11 hours out of phase with the ASPA
     record -- close to antiphase for diurnal variables.

  2. Training rows are kept unless the TARGET is missing. The original applied
     a bare .dropna() across all ~40 columns, so a single NaN anywhere in a row
     discarded it. XGBoost and LightGBM handle missing features natively.

  3. Training uses valid data on BOTH sides of a bounded gap. The original cut
     at the gap start, discarding every observation after the sensor recovered.

  4. Coordinates and leading blank rows are found from the data rather than
     hardcoded indices.
"""

from __future__ import annotations

import argparse
import sys
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


# ==============================================================================
# Utilities
# ==============================================================================

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column except TIMESTAMP to numeric, in place on a copy."""
    df = df.copy()
    for col in df.columns:
        if col != "TIMESTAMP":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def circular_weight_components(wd_deg, w):
    """Weight the sin/cos components of a direction, never the raw degrees.

    Degrees wrap at 0/360, so weighting or averaging them directly produces
    nonsense near north.
    """
    wd_rad = np.deg2rad(pd.to_numeric(wd_deg, errors="coerce"))
    return np.sin(wd_rad) * w, np.cos(wd_rad) * w


def first_valid(series: pd.Series):
    """First non-null value, or raise with a useful message."""
    s = series.dropna()
    if s.empty:
        raise ValueError(f"Column {series.name!r} has no valid values.")
    return s.iloc[0]


# ==============================================================================
# Station loading
# ==============================================================================

STATION_COLUMNS = ["TIMESTAMP", "LAT", "LON", "PTemp_C_Max", "AirTF_Avg",
                   "SlrW_Avg", "SlrMJ_Tot", "RH", "Rain_in_Tot"]


def find_station_file(station: str) -> Path | None:
    """Locate a station's cleaned CSV by prefix, tolerating naming variations.

    Synoptic station files live in the same directory and can start with the
    same letters in principle, so anything matching a synoptic station ID is
    excluded.
    """
    matches = [p for p in sorted(cfg.CLEANED_RAW_DIR.glob(
        cfg.STATION_FILE_PATTERN.format(station=station)))
        if not any(p.name.startswith(s) for s in cfg.SYNOPTIC_STATIONS)]

    if not matches:
        return None
    chosen = max(matches, key=lambda p: p.stat().st_mtime)
    if len(matches) > 1:
        others = ", ".join(p.name for p in matches if p != chosen)
        print(f"  {station}: using {chosen.name} (also found: {others})")
    else:
        print(f"  {station}: {chosen.name}")
    return chosen


def load_stations() -> pd.DataFrame:
    """Load all ASPA stations and outer-join them on TIMESTAMP.

    Each station's columns are suffixed with its name, so Aasu's relative
    humidity becomes RH_Aasu.
    """
    if not cfg.CLEANED_RAW_DIR.exists():
        raise FileNotFoundError(
            f"Cleaned data directory not found:\n    {cfg.CLEANED_RAW_DIR}\n"
            f"Run `python pipeline.py --paths` to see all resolved locations."
        )

    frames = []
    for name in cfg.STATIONS:
        path = find_station_file(name)
        if path is None:
            print(f"  WARNING: no file matching {name}*.csv in "
                  f"{cfg.CLEANED_RAW_DIR}; skipping station")
            continue

        df = pd.read_csv(path, low_memory=False)
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
        df = df.drop(columns=[c for c in cfg.ALWAYS_DROP if c in df.columns])
        df = to_numeric(df)

        # Duplicate timestamps multiply rows in the outer join below, silently
        # inflating the frame. Collapse them by averaging.
        n_dupe = df["TIMESTAMP"].duplicated().sum()
        if n_dupe:
            print(f"    {n_dupe} duplicate timestamp(s) collapsed by mean")
            df = df.groupby("TIMESTAMP", as_index=False).mean(numeric_only=True)

        missing = [c for c in STATION_COLUMNS if c not in df.columns]
        if missing:
            print(f"  WARNING: {name} missing {missing}; skipping station")
            continue

        renamed = df[STATION_COLUMNS].rename(
            columns={c: f"{c}_{name}" for c in STATION_COLUMNS if c != "TIMESTAMP"}
        )
        frames.append(renamed)

    if not frames:
        raise RuntimeError("No station files could be loaded.")

    combined = reduce(
        lambda l, r: pd.merge(l, r, on="TIMESTAMP", how="outer"), frames
    ).sort_values("TIMESTAMP").reset_index(drop=True)

    # The original used a hardcoded .iloc[16:] to skip leading rows from before
    # all stations were reporting. Find them instead: drop leading rows where
    # every non-timestamp column is null.
    data_cols = [c for c in combined.columns if c != "TIMESTAMP"]
    all_null = combined[data_cols].isna().all(axis=1)
    first_real = all_null.idxmin() if all_null.any() else 0
    if first_real > 0:
        print(f"  dropped {first_real} leading all-null rows "
              f"(original hardcoded 16)")
        combined = combined.iloc[first_real:].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Neighbour coverage. The stations start years apart -- Aasu from 2017-04
    # and Afono from 2017-08, but Poloa and Vaipito only from mid-2019 -- so
    # early training rows have fewer neighbours reporting than a 2020-2022 gap
    # row does. Recording how many neighbours are present lets the model
    # condition on that regime explicitly instead of inferring it from a
    # pattern of missing columns.
    # ------------------------------------------------------------------
    present = pd.DataFrame(index=combined.index)
    for s in cfg.STATIONS:
        cols = [f"{v}_{s}" for v in cfg.MET_VARIABLES
                if f"{v}_{s}" in combined.columns]
        if cols:
            present[s] = combined[cols].notna().any(axis=1)
    if len(present.columns):
        combined["stations_reporting"] = present.sum(axis=1).astype("int8")
        by_count = combined["stations_reporting"].value_counts().sort_index()
        print("  stations reporting per row: " +
              ", ".join(f"{k}:{v:,}" for k, v in by_count.items()))

    print(f"  combined station frame: {combined.shape}")
    return combined


# ==============================================================================
# Synoptic loading
# ==============================================================================

SYNOPTIC_VARS = ["air_temp_set_1", "relative_humidity_set_1",
                 "wind_speed_set_1", "wind_direction_set_1"]


def find_synoptic_files() -> list[Path]:
    """Locate one file per synoptic station, tolerating either naming scheme.

    Matches both the manual web-app export (NSTU.2022-12-31.csv) and the API
    download script's output (NSTU_2017-2024.csv). Prefers the most recently
    modified when several match.
    """
    found = []
    search_dirs = [cfg.SYNOPTIC_DIR, cfg.CLEANED_RAW_DIR]

    for stid in cfg.SYNOPTIC_STATIONS:
        matches = []
        for d in search_dirs:
            if d.exists():
                matches += sorted(d.glob(f"{stid}*.csv"))
        if not matches:
            print(f"  WARNING: no file found for synoptic station {stid}")
            continue
        chosen = max(matches, key=lambda p: p.stat().st_mtime)
        if len(matches) > 1:
            print(f"  {stid}: {len(matches)} files found, using {chosen.name}")
        found.append(chosen)

    return found


def _has_units_row(path: Path) -> bool:
    """Manual web-app exports carry a units row under the header; the API
    output does not. Detect rather than assume, so both formats load."""
    head = pd.read_csv(path, nrows=2)
    if len(head) == 0:
        return False
    first = head.iloc[0]
    # A units row is non-numeric in columns that should hold measurements.
    for col in SYNOPTIC_VARS:
        if col in head.columns:
            val = first[col]
            if isinstance(val, str) and pd.to_numeric(val, errors="coerce") is not None:
                if pd.isna(pd.to_numeric(val, errors="coerce")) and val.strip() != "":
                    return True
    return False


def load_synoptic(path: Path) -> pd.DataFrame:
    """Load one synoptic file, convert to local time, and upsample to 15 min."""
    skiprows = [1] if _has_units_row(path) else None
    df = pd.read_csv(path, skiprows=skiprows)

    # --- The timezone fix. ---
    # The source timestamps carry a -1100 offset. Parsing with utc=True and
    # then tz_localize(None) yields naive UTC, which is 11 hours ahead of the
    # naive local time the ASPA loggers record. Convert to local first.
    ts = pd.to_datetime(df["TIMESTAMP"], utc=True, errors="coerce")
    df["TIMESTAMP"] = ts.dt.tz_convert(cfg.LOCAL_TZ).dt.tz_localize(None)
    df = df[df["TIMESTAMP"].notna()].copy()
    df["TIMESTAMP"] = df["TIMESTAMP"].dt.round("15min")

    lat = first_valid(df["LAT"]) if "LAT" in df.columns else None
    lon = first_valid(df["LON"]) if "LON" in df.columns else None
    elev = first_valid(df["Elevation"]) if "Elevation" in df.columns else np.nan

    keep = ["TIMESTAMP"] + [c for c in SYNOPTIC_VARS if c in df.columns]
    df = df[keep]
    for c in SYNOPTIC_VARS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.groupby("TIMESTAMP", as_index=False).mean(numeric_only=True)
    temp = df.set_index("TIMESTAMP")

    # Wind direction is circular: interpolate the speed-scaled u/v components
    # and recover the angle, rather than interpolating degrees across the
    # 0/360 wrap.
    if {"wind_direction_set_1", "wind_speed_set_1"} <= set(temp.columns):
        wd_rad = np.deg2rad(temp["wind_direction_set_1"])
        u = temp["wind_speed_set_1"] * np.sin(wd_rad)
        v = temp["wind_speed_set_1"] * np.cos(wd_rad)
        try:
            u_i = u.resample("15min").interpolate("akima")
            v_i = v.resample("15min").interpolate("akima")
            wd_interp = (np.rad2deg(np.arctan2(u_i, v_i)) + 360) % 360
        except Exception:
            wd_interp = (temp["wind_direction_set_1"]
                         .resample("15min").interpolate("nearest"))
    else:
        wd_interp = None

    scalar_cols = [c for c in ["air_temp_set_1", "relative_humidity_set_1",
                               "wind_speed_set_1"] if c in temp.columns]
    out = temp[scalar_cols].resample("15min").interpolate("akima").reset_index()

    if wd_interp is not None:
        out["wind_direction_set_1"] = wd_interp.reindex(
            pd.DatetimeIndex(out["TIMESTAMP"])).values

    # Elevation is a station constant; the original akima-interpolated it,
    # which is harmless but meaningless. Assign it directly.
    out["Elevation"] = elev
    out["LAT"] = lat
    out["LON"] = lon
    return out


# ==============================================================================
# Feature construction
# ==============================================================================

def add_neighbor_features(df: pd.DataFrame, target_station: str,
                          power: int) -> pd.DataFrame:
    """Add inverse-distance-weighted copies of every neighbour station's
    variables. The target station's own columns are left unweighted -- they
    are at distance zero and are used directly."""
    df = df.copy()
    tgt_lat = first_valid(df[f"LAT_{target_station}"])
    tgt_lon = first_valid(df[f"LON_{target_station}"])

    neighbors = [s for s in cfg.STATIONS if s != target_station]
    for s in neighbors:
        if f"LAT_{s}" not in df.columns:
            continue
        lat_s = first_valid(df[f"LAT_{s}"])
        lon_s = first_valid(df[f"LON_{s}"])
        dist = max(haversine_km(lat_s, lon_s, tgt_lat, tgt_lon), 0.1)
        w = 1.0 / (dist ** power)

        for var in cfg.MET_VARIABLES + ["PTemp_C_Max"]:
            col = f"{var}_{s}"
            if col in df.columns:
                df[f"weighted_{var}_{s}"] = df[col] * w

    return df


def add_synoptic_features(df: pd.DataFrame, target_station: str,
                          synoptic: list[pd.DataFrame], power: int) -> pd.DataFrame:
    """Merge each synoptic station in as distance-weighted features."""
    df = df.copy()
    tgt_lat = first_valid(df[f"LAT_{target_station}"])
    tgt_lon = first_valid(df[f"LON_{target_station}"])

    for i, syn in enumerate(synoptic):
        if syn is None or syn.empty:
            continue
        merged = df[["TIMESTAMP"]].merge(syn, on="TIMESTAMP", how="left")

        dist = max(haversine_km(first_valid(syn["LAT"]), first_valid(syn["LON"]),
                                tgt_lat, tgt_lon), 0.1)
        w = 1.0 / (dist ** power)

        if "wind_direction_set_1" in merged.columns:
            sin_w, cos_w = circular_weight_components(
                merged["wind_direction_set_1"], w)
            df[f"syn{i}_wind_dir_sin_w"] = sin_w.values
            df[f"syn{i}_wind_dir_cos_w"] = cos_w.values
        for src, dst in [("wind_speed_set_1", "wind_speed"),
                         ("air_temp_set_1", "air_temp"),
                         ("relative_humidity_set_1", "rel_humidity")]:
            if src in merged.columns:
                df[f"syn{i}_{dst}_w"] = merged[src].values * w

        df[f"syn{i}_elevation"] = first_valid(syn["Elevation"]) \
            if "Elevation" in syn.columns else np.nan

    return df


def drop_intermediates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove coordinate columns, which are constants per station and carry no
    per-timestamp information once distances are computed."""
    drop = [c for c in df.columns
            if c.startswith("LAT_") or c.startswith("LON_")]
    return df.drop(columns=drop)


# ==============================================================================
# Train / prediction split
# ==============================================================================

def enforce_feature_availability(df: pd.DataFrame, run: dict,
                                 verbose: bool = True):
    """Drop features that do not exist during the gap being filled.

    Availability is measured on the prediction rows only -- those are the rows
    the model will actually face. A feature present throughout training but
    absent over the gap contributes nothing at prediction time and actively
    displaces usable splits, so it is removed from BOTH sets rather than left
    for the booster's missing-value handling to absorb.

    Returns the trimmed frame and a report of what was dropped and why.
    """
    target = run["target_column"]
    start = pd.to_datetime(run["start"])
    end = pd.to_datetime(run["end"]) if run["end"] else None

    in_gap = df["TIMESTAMP"] >= start
    if end is not None:
        in_gap &= df["TIMESTAMP"] <= end
    gap_rows = df[in_gap]

    if gap_rows.empty:
        raise ValueError(f"No rows fall inside the gap for {target}.")

    protected = {"TIMESTAMP", target}
    features = [c for c in df.columns if c not in protected]

    availability = gap_rows[features].notna().mean()

    # Manual drops from config always apply, plus their weighted derivatives.
    manual = set()
    for col in run["extra_drops"]:
        manual |= {c for c in features
                   if c == col or c == f"weighted_{col}"}

    auto = {c for c in features
            if availability[c] < cfg.MIN_PRED_AVAILABILITY} - manual

    if verbose:
        print(f"  feature availability during gap "
              f"(threshold {cfg.MIN_PRED_AVAILABILITY:.0%})")
        print(f"    kept   : {len(features) - len(manual | auto)}")
        if manual:
            print(f"    manual drops ({len(manual)}):")
            for c in sorted(manual):
                print(f"        {c:<34} {availability[c]:6.1%} available")
        if auto:
            print(f"    below threshold ({len(auto)}):")
            for c in sorted(auto, key=lambda x: availability[x]):
                print(f"        {c:<34} {availability[c]:6.1%} available")
        # Features that are fine in the gap but thin overall are worth seeing.
        marginal = [c for c in features
                    if c not in manual | auto
                    and availability[c] < 0.9]
        if marginal:
            print(f"    kept but patchy ({len(marginal)}):")
            for c in sorted(marginal, key=lambda x: availability[x]):
                print(f"        {c:<34} {availability[c]:6.1%} available")

    return df.drop(columns=sorted(manual | auto)), sorted(manual | auto)


def split_train_pred(df: pd.DataFrame, run: dict):
    """Split into training rows and gap rows.

    Training takes every row OUTSIDE the gap where the target is observed --
    including rows after the sensor recovered, for bounded gaps. Feature NaNs
    are left in place for the boosters to handle.
    """
    target = run["target_column"]
    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not in frame.")

    start = pd.to_datetime(run["start"])
    end = pd.to_datetime(run["end"]) if run["end"] else None

    in_gap = df["TIMESTAMP"] >= start
    if end is not None:
        in_gap &= df["TIMESTAMP"] <= end

    outside = df[~in_gap]
    n_before = len(outside)
    train = outside.dropna(subset=[target])
    pred = df[in_gap].copy()

    # Row-level coverage: drop training rows too sparse to be informative.
    n_target_ok = len(train)
    feature_cols = [c for c in train.columns
                    if c not in {"TIMESTAMP", target}]
    if feature_cols and cfg.MIN_TRAIN_FEATURE_COVERAGE > 0:
        coverage = train[feature_cols].notna().mean(axis=1)
        train = train[coverage >= cfg.MIN_TRAIN_FEATURE_COVERAGE]

    # Diagnostic: what the original bare .dropna() would have kept.
    would_have_kept = len(outside.dropna())
    print(f"  rows outside gap        : {n_before:,}")
    print(f"  training rows (target ok): {n_target_ok:,}")
    dropped_sparse = n_target_ok - len(train)
    if dropped_sparse:
        print(f"  dropped, under {cfg.MIN_TRAIN_FEATURE_COVERAGE:.0%} feature "
              f"coverage: {dropped_sparse:,}")
    print(f"  training rows (final)    : {len(train):,}")
    print(f"  original bare .dropna()  : {would_have_kept:,} "
          f"({would_have_kept / max(n_before, 1):.1%} of available)")
    if len(train):
        print(f"  train span              : {train['TIMESTAMP'].min()} "
              f"-> {train['TIMESTAMP'].max()}")
    print(f"  prediction rows         : {len(pred):,}")
    if len(pred):
        print(f"  pred span               : {pred['TIMESTAMP'].min()} "
              f"-> {pred['TIMESTAMP'].max()}")

    if run["bounded_gap"] and len(train):
        after = (train["TIMESTAMP"] > (end or start)).sum()
        print(f"  of which post-gap       : {after:,} "
              f"(discarded by the original script)")

    return train, pred


# ==============================================================================
# Orchestration
# ==============================================================================

def build_run(combined: pd.DataFrame, synoptic: list[pd.DataFrame],
              station: str, variable: str, window: int = 0,
              dry_run: bool = False):
    run = cfg.run_config(station, variable, window)
    tag = f" ({run['label']})" if run["label"] else ""
    print(f"\n{run['target_column']}{tag}  [{run['scope']}, "
          f"{'bounded' if run['bounded_gap'] else 'open-ended'}]")
    print(f"  gap: {run['start']} -> {run['end'] or 'end of record'}")

    if run["scope"] != "neighbors_and_synoptic":
        print("  scope not yet implemented in this module; skipping")
        return None, None

    df = add_neighbor_features(combined, station, cfg.IDW_POWER)
    df = add_synoptic_features(df, station, synoptic, cfg.IDW_POWER)
    df = drop_intermediates(df)

    # Features must exist during the gap, not merely during training.
    df, dropped = enforce_feature_availability(df, run)

    train, pred = split_train_pred(df, run)

    if not dry_run:
        cfg.MODEL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        train_path = cfg.MODEL_INPUT_DIR / f"{run['output_stem']}_train.csv"
        pred_path = cfg.MODEL_INPUT_DIR / f"{run['output_stem']}_pred.csv"
        train.to_csv(train_path, index=False)
        pred.to_csv(pred_path, index=False)
        print(f"  wrote {train_path.name} and {pred_path.name}")

    return train, pred


def report_paths() -> None:
    """Print every resolved directory and what is actually in it.

    First thing to run when a file cannot be found: it distinguishes a wrong
    directory from a wrong filename.
    """
    print("Resolved paths\n")
    for label, path in [
        ("repo root", cfg.REPO_ROOT),
        ("ASCDP", cfg.ASCDP_DIR),
        ("Data Cleaning", cfg.DATA_CLEANING_DIR),
        ("Cleaned Raw Data", cfg.CLEANED_RAW_DIR),
        ("Model Input Data", cfg.MODEL_INPUT_DIR),
        ("Synoptic", cfg.SYNOPTIC_DIR),
    ]:
        mark = "ok     " if path.exists() else "MISSING"
        print(f"  [{mark}] {label:<18} {path}")

    if cfg.CLEANED_RAW_DIR.exists():
        print(f"\nCSV files in {cfg.CLEANED_RAW_DIR.name}:")
        files = sorted(cfg.CLEANED_RAW_DIR.glob("*.csv"))
        for f in files[:40]:
            print(f"    {f.name}")
        if len(files) > 40:
            print(f"    ... and {len(files) - 40} more")
        if not files:
            print("    (none)")

        print("\nStation file matches:")
        for s in cfg.STATIONS:
            m = [p.name for p in cfg.CLEANED_RAW_DIR.glob(
                cfg.STATION_FILE_PATTERN.format(station=s))]
            print(f"    {s:<10} {m if m else 'NO MATCH'}")

    print("\nSynoptic file matches:")
    for stid in cfg.SYNOPTIC_STATIONS:
        m = []
        for d in [cfg.SYNOPTIC_DIR, cfg.CLEANED_RAW_DIR]:
            if d.exists():
                m += [p.name for p in d.glob(f"{stid}*.csv")]
        print(f"    {stid:<10} {m if m else 'NO MATCH'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station")
    p.add_argument("--variable")
    p.add_argument("--all", action="store_true",
                   help="Run every configured combination.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report shapes without writing files.")
    p.add_argument("--paths", action="store_true",
                   help="Show resolved directories and matching files, then exit.")
    args = p.parse_args()

    if args.paths:
        report_paths()
        return

    if not args.all and not (args.station and args.variable):
        p.error("Give --station and --variable, or --all (or --paths).")

    print("Loading stations")
    combined = load_stations()

    print("Loading synoptic")
    synoptic = [load_synoptic(p_) for p_ in find_synoptic_files()]
    print(f"  {len(synoptic)} synoptic station(s) loaded")

    if args.all:
        runs = cfg.all_runs()
    else:
        n = len(cfg.GAP_WINDOWS.get((args.station, args.variable), []))
        runs = [(args.station, args.variable, i) for i in range(n)]
        if not runs:
            p.error(f"No gaps configured for "
                    f"({args.station}, {args.variable}).")

    met_runs = [r for r in runs
                if cfg.scope_for(r[1]) == "neighbors_and_synoptic"]

    if args.all:
        print(f"\n{len(met_runs)} met run(s) of {len(runs)} configured")

    for station, variable, window in met_runs:
        try:
            build_run(combined, synoptic, station, variable, window,
                      args.dry_run)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()