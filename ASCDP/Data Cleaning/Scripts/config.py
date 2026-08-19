#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central configuration for the ASPA gap-filling pipeline.

Everything that differs between runs lives here. No other module should
contain a station name, a variable name, a date, or a file path.

The run key is always the tuple (station, variable). Given a key, this module
answers four questions:

    1. When is the gap?              -> GAP_WINDOWS
    2. What features are allowed?    -> FEATURE_SCOPE (via scope_for)
    3. What is the target's type?    -> TARGET_TYPE  (scalar vs circular)
    4. What else must be dropped?    -> EXTRA_FEATURE_DROPS

Add a new run by adding one entry to GAP_WINDOWS. Nothing else needs editing.
"""

from pathlib import Path

# ==============================================================================
# Paths -- resolved from this file's location, never absolute
# ==============================================================================

# config.py is expected at <repo>/ASCDP/Data Cleaning/Scripts/config.py
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_CLEANING_DIR = SCRIPT_DIR.parent
ASCDP_DIR = DATA_CLEANING_DIR.parent
REPO_ROOT = ASCDP_DIR.parent

CLEANED_RAW_DIR = DATA_CLEANING_DIR / "Cleaned Raw Data"
MODEL_INPUT_DIR = DATA_CLEANING_DIR / "Cleaned Model Input Data"
SYNOPTIC_DIR = ASCDP_DIR / "Raw Input Data" / "Synoptic"

# ==============================================================================
# Stations
# ==============================================================================

STATIONS = ["Aasu", "Poloa", "Afono", "Vaipito"]

# Station files are matched by prefix rather than an exact name, so the
# pipeline tolerates the various cleaned-file naming schemes in use
# (Aasu_cleaned.csv, Aasu_ALL_15min_data_cleaned.csv, ...). Where several
# match, the most recently modified wins and the choice is reported.
STATION_FILE_PATTERN = "{station}*.csv"

# Synoptic sources. Filenames are matched by prefix so the pipeline works with
# either the manual web-app export (NSTU.2022-12-31.csv) or the API download
# script's output (NSTU_2017-2024.csv).
SYNOPTIC_STATIONS = ["NSTU", "SFGP6"]

# Observations are recorded in local Samoa time. The ASPA loggers write naive
# local timestamps, so synoptic timestamps must be CONVERTED to local before
# the tz marker is stripped -- not simply localized to UTC.
LOCAL_TZ = "Pacific/Pago_Pago"

# ==============================================================================
# Variables
# ==============================================================================

# Columns present in the raw station files that are never used as features
# by any run. Note this list is deliberately short: wind columns are NOT
# dropped globally, because the wind runs need them.
ALWAYS_DROP = ["RECORD", "BattV_Avg"]

MET_VARIABLES = ["AirTF_Avg", "RH", "SlrW_Avg", "SlrMJ_Tot", "Rain_in_Tot"]
WIND_VARIABLES = ["WS_mph_S_WVT", "WindDir_D1_WVT", "WindDir_SD1_WVT"]

# Non-target columns carried along for identification / weighting.
COORD_COLUMNS = ["LAT", "LON"]

# ------------------------------------------------------------------
# Feature scope: which stations may contribute features.
#
# 'neighbors_and_synoptic'   -- the other three ASPA stations (IDW weighted)
#                               plus the target station's own other sensors,
#                               plus synoptic. Used for met variables, where
#                               neighbour stations have good coverage.
#
# 'own_station_and_synoptic' -- only the target station's other sensors plus
#                               synoptic. Used for wind, where the wind
#                               sensors at the other stations have too much
#                               missing data to be useful contributors.
# ------------------------------------------------------------------

FEATURE_SCOPE = {
    "neighbors_and_synoptic": MET_VARIABLES,
    "own_station_and_synoptic": WIND_VARIABLES,
}

def scope_for(variable: str) -> str:
    for scope, variables in FEATURE_SCOPE.items():
        if variable in variables:
            return scope
    raise KeyError(f"No feature scope defined for variable {variable!r}")

# ------------------------------------------------------------------
# Target type: how the variable must be modelled.
#
# 'scalar'   -- one regressor, predict the value directly.
#
# 'circular' -- the 0/360 wraparound makes direct regression invalid. Fit two
#               regressors on sin(theta) and cos(theta), then recombine with
#               arctan2. Evaluate with angular difference, NOT plain RMSE.
#
# WindDir_SD1_WVT is intentionally 'scalar': it is a measure of spread with a
# natural zero, bounded roughly 0-180, and has no wraparound. It does not need
# vector decomposition.
# ------------------------------------------------------------------

TARGET_TYPE = {
    "AirTF_Avg": "scalar",
    "RH": "scalar",
    "SlrW_Avg": "scalar",
    "SlrMJ_Tot": "scalar",
    "Rain_in_Tot": "scalar",
    "WS_mph_S_WVT": "scalar",
    "WindDir_SD1_WVT": "scalar",
    "WindDir_D1_WVT": "circular",
}

# Sub-model labels for circular targets. These are DERIVED FROM THE TARGET and
# must never appear in the feature matrix -- a model handed WindDir_sin while
# predicting WindDir_sin will score near-perfectly and mean nothing. The
# pipeline emits them as labels; any code building X must exclude them.
CIRCULAR_LABELS = ["WindDir_sin", "WindDir_cos"]

def label_columns(variable: str, station: str) -> list:
    """Columns that are labels for this run, not features."""
    cols = [f"{variable}_{station}"]
    if TARGET_TYPE[variable] == "circular":
        cols += CIRCULAR_LABELS
    return cols

# Own-station sensors offered as features to the wind runs. The anemometer is
# dead over the prediction period, so the usable signal at the target station
# is its remaining met sensors.
OWN_STATION_FEATURES = ["PTemp_C_Max", "AirTF_Avg", "RH", "SlrW_Avg", "SlrMJ_Tot"]

# Below this wind speed the direction vane wanders and recorded direction is
# effectively noise. Observations under this threshold are excluded when
# scoring direction predictions. Units match WS_mph_S_WVT after metric
# conversion (m/s). State this threshold in the methods section.
CALM_THRESHOLD_MS = 0.5

# ==============================================================================
# Gap windows -- the source of truth for what each run fills
#
# `start`  first timestamp of the sensor failure.
# `end`    last timestamp of the failure, or None if the sensor never
#          recovered (all wind sensors are in this category).
#
# Why `end` matters: where a gap is BOUNDED, valid observations exist on both
# sides and training uses all of them -- gap-filling is interpolation, not
# forecasting, so post-gap data is legitimate and excluding it costs accuracy
# for no methodological benefit. Where a gap is OPEN-ENDED there is no other
# side, training is necessarily one-sided, and the task is closer to
# extrapolation. That distinction drives how each run must be validated
# (see HOLDOUT_STRATEGY below), so it is recorded here rather than inferred.
#
# `extra_drops` lists feature columns removed for this run because they are
# themselves broken over an overlapping period.
# ==============================================================================

GAP_WINDOWS = {
    # ---------------- Met variables ----------------
    ("Aasu", "RH"): [
        {"start": "2020-03-09 08:45:00", "end": "2021-04-10 22:00:00",
         "extra_drops": ["AirTF_Avg_Aasu"], "label": "sensor"},
        {"start": "2018-01-29 14:45:00", "end": "2018-04-22 04:00:00",
         "extra_drops": [], "label": "station_outage"},
    ],
    ("Aasu", "AirTF_Avg"): [
        {"start": "2020-03-09 08:45:00", "end": "2021-04-10 22:00:00",
         "extra_drops": ["RH_Aasu"], "label": "sensor"},
        {"start": "2018-01-29 14:45:00", "end": "2018-04-22 04:00:00",
         "extra_drops": [], "label": "station_outage"},
    ],
    ("Aasu", "Rain_in_Tot"): [
        {"start": "2020-04-14 00:15:00", "end": "2022-03-25 13:45:00",
         "extra_drops": ["RH_Aasu", "AirTF_Avg_Aasu"], "label": "sensor"},
        {"start": "2018-01-29 14:45:00", "end": "2018-04-22 04:00:00",
         "extra_drops": [], "label": "station_outage"},
    ],
    # Aasu solar has no sensor-specific failure; only the 2018 station outage.
    ("Aasu", "SlrW_Avg"): [
        {"start": "2018-01-29 14:45:00", "end": "2018-04-22 04:00:00",
         "extra_drops": [], "label": "station_outage"},
    ],
    ("Aasu", "SlrMJ_Tot"): [
        {"start": "2018-01-29 14:45:00", "end": "2018-04-22 04:00:00",
         "extra_drops": [], "label": "station_outage"},
    ],

    # Poloa and Vaipito pyranometers failed within 24 hours of each other in
    # March 2022, so neither can serve as a feature for the other. The
    # availability check drops them automatically; they are not listed in
    # extra_drops so that the measurement, rather than an assumption, is what
    # removes them.
    ("Poloa", "SlrW_Avg"): [
        {"start": "2022-03-22 06:00:00", "end": None,
         "extra_drops": ["SlrMJ_Tot_Poloa"], "label": "sensor"},
    ],
    ("Poloa", "SlrMJ_Tot"): [
        {"start": "2022-03-22 06:00:00", "end": None,
         "extra_drops": ["SlrW_Avg_Poloa"], "label": "sensor"},
    ],
    ("Vaipito", "SlrW_Avg"): [
        {"start": "2022-03-23 10:30:00", "end": None,
         "extra_drops": ["SlrMJ_Tot_Vaipito"], "label": "sensor"},
    ],
    ("Vaipito", "SlrMJ_Tot"): [
        {"start": "2022-03-23 10:30:00", "end": None,
         "extra_drops": ["SlrW_Avg_Vaipito"], "label": "sensor"},
    ],

    # ---------------- Wind variables ----------------
    # Aasu and Afono anemometers were restored in late April 2022, leaving
    # roughly three months of post-recovery observations: those runs are
    # BOUNDED and can be scored against real data. Poloa and Vaipito never
    # recovered within the record.
    ("Aasu", "WS_mph_S_WVT"):    [{"start": "2019-08-27 20:30:00", "end": "2022-04-28 12:45:00", "extra_drops": [], "label": "sensor"}],
    ("Aasu", "WindDir_D1_WVT"):  [{"start": "2019-08-27 20:30:00", "end": "2022-04-28 12:45:00", "extra_drops": [], "label": "sensor"}],
    ("Aasu", "WindDir_SD1_WVT"): [{"start": "2019-08-27 20:30:00", "end": "2022-04-28 12:45:00", "extra_drops": [], "label": "sensor"}],

    ("Afono", "WS_mph_S_WVT"):    [{"start": "2019-08-16 14:00:00", "end": "2022-04-26 12:45:00", "extra_drops": [], "label": "sensor"}],
    ("Afono", "WindDir_D1_WVT"):  [{"start": "2019-08-16 14:00:00", "end": "2022-04-26 12:45:00", "extra_drops": [], "label": "sensor"}],
    ("Afono", "WindDir_SD1_WVT"): [{"start": "2019-08-16 14:00:00", "end": "2022-04-26 12:45:00", "extra_drops": [], "label": "sensor"}],

    ("Vaipito", "WS_mph_S_WVT"):    [{"start": "2020-10-14 20:00:00", "end": None, "extra_drops": [], "label": "sensor"}],
    ("Vaipito", "WindDir_D1_WVT"):  [{"start": "2020-10-14 20:00:00", "end": None, "extra_drops": [], "label": "sensor"}],
    ("Vaipito", "WindDir_SD1_WVT"): [{"start": "2020-10-14 20:00:00", "end": None, "extra_drops": [], "label": "sensor"}],

    ("Poloa", "WS_mph_S_WVT"):    [{"start": "2020-06-20 05:15:00", "end": None, "extra_drops": [], "label": "sensor"}],
    ("Poloa", "WindDir_D1_WVT"):  [{"start": "2020-06-20 05:15:00", "end": None, "extra_drops": [], "label": "sensor"}],
    ("Poloa", "WindDir_SD1_WVT"): [{"start": "2020-06-20 05:15:00", "end": None, "extra_drops": [], "label": "sensor"}],
}

# ==============================================================================
# Modelling
# ==============================================================================

IDW_POWER = 2

# ------------------------------------------------------------------
# Feature availability during the prediction window.
#
# A feature is only useful if it exists when the gap needs filling. If a
# column is present throughout training but absent during the gap, the model
# builds splits that lean on it and then, at prediction time, sends every row
# down the "missing" branch -- a branch fitted on whatever unrepresentative
# slice of training rows happened to lack that feature. That is worse than
# excluding the feature outright, because it displaces splits the model could
# have made on data that is actually there.
#
# Any feature observed in fewer than this fraction of prediction rows is
# dropped from both the training and prediction sets. Entries in a run's
# `extra_drops` are removed as well, regardless of measured availability.
# ------------------------------------------------------------------
MIN_PRED_AVAILABILITY = 0.5

# ------------------------------------------------------------------
# Minimum feature coverage for a TRAINING row.
#
# The column-level check above removes features absent during the gap. This
# is the same idea one level down: a row that carries a valid target but
# almost no surviving features teaches a relationship built on whatever few
# inputs it has, which then competes with the richer relationship that
# actually applies over the gap.
#
# The case that motivates it: the Aasu 2018 station-outage runs drop every
# Aasu, Poloa and Vaipito feature, leaving Afono and synoptic. But the record
# starts 2017-04-25 and Afono only begins 2017-08-24, so about four months of
# training rows have a target and synoptic data alone.
#
# A row is kept when at least this fraction of the retained feature columns
# are observed. Set to 0.0 to disable.
# ------------------------------------------------------------------
MIN_TRAIN_FEATURE_COVERAGE = 0.0

# Holdout construction. A random 20% of 15-minute rows would be dishonest:
# predicting an isolated missing sample from its neighbours 15 minutes either
# side is far easier than filling a months-long hole, and would overstate
# performance badly. Instead, hold out SYNTHETIC GAPS -- contiguous blocks of
# valid data, of comparable length and season to the real gap being filled --
# and score there.
#
# This matters most for the open-ended wind runs, where the real gap can never
# be scored against ground truth at all: the sensor never recovered, so
# reported wind accuracy rests entirely on synthetic gaps. Those blocks should
# be long, and placed across seasons, or the numbers will not survive review.
HOLDOUT_STRATEGY = "synthetic_gap"   # 'synthetic_gap' | 'random'
SYNTHETIC_GAP_COUNT = 5
RANDOM_SEED = 42

# ==============================================================================
# Derived helpers
# ==============================================================================

def all_runs():
    """Every (station, variable, window_index) the pipeline knows how to run.

    A station-variable may have several gaps -- Aasu's sensor failures plus the
    2018 whole-station outage, for instance -- so the window index is part of
    the run identity.
    """
    runs = []
    for (station, variable), windows in GAP_WINDOWS.items():
        for i in range(len(windows)):
            runs.append((station, variable, i))
    return sorted(runs)


def run_config(station: str, variable: str, window: int = 0) -> dict:
    """Everything needed to execute one run, assembled from the tables above."""
    key = (station, variable)
    if key not in GAP_WINDOWS:
        raise KeyError(
            f"No gap window configured for {key}. "
            f"Known: {sorted(GAP_WINDOWS.keys())}"
        )
    windows = GAP_WINDOWS[key]
    if window >= len(windows):
        raise IndexError(
            f"{key} has {len(windows)} window(s); asked for index {window}."
        )

    w = windows[window]
    bounded = w["end"] is not None
    label = w.get("label")

    # Only disambiguate the filename when a variable has more than one gap.
    stem = f"{station.lower()}_{variable}"
    if len(windows) > 1 and label:
        stem = f"{stem}_{label}"

    return {
        "station": station,
        "variable": variable,
        "window_index": window,
        "label": label,
        "target_column": f"{variable}_{station}",
        "label_columns": label_columns(variable, station),
        "start": w["start"],
        "end": w["end"],
        "bounded_gap": bounded,
        # Bounded gaps have valid data after the gap that belongs in training.
        # Open-ended failures do not, so training is one-sided by necessity.
        "train_both_sides": bounded,
        "extra_drops": list(w.get("extra_drops", [])),
        "scope": scope_for(variable),
        "target_type": TARGET_TYPE[variable],
        "output_stem": stem,
    }