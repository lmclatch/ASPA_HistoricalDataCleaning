#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Cleaning Pipeline
@author: lizamclatchy
"""

import re
import pandas as pd
import numpy as np
from functools import reduce

# ==============================================================================
# Configuration
# ==============================================================================

station_files = {
    'Aasu':    '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Aasu_ALL_15min_data_cleaned.csv',
    'Poloa':   '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Poloa_ALL_15min_data_cleaned.csv',
    'Afono':   '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Afono_ALL_15min_data_cleaned.csv',
    'Vaipito': '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Vaipito_ALL_15min_data_cleaned.csv',
}

cols_to_drop     = ['WS_mph_S_WVT', 'WindDir_D1_WVT', 'WindDir_SD1_WVT', 'RECORD', 'BattV_Avg']
required_columns = ['TIMESTAMP', 'LAT', 'LON', 'PTemp_C_Max', 'AirTF_Avg',
                    'SlrW_Avg', 'SlrMJ_Tot', 'RH', 'Rain_in_Tot']

synoptic_files = [
    '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/NSTU.2022-12-31.csv',
    '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/SFGP6.2022-12-31.csv',
]

# ==============================================================================
# Utilities
# ==============================================================================

def convert_to_numeric(df):
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def circular_weight_components(wd_deg_series, w):
    """Correct circular handling: weight sin/cos components, not raw degrees."""
    wd     = pd.to_numeric(wd_deg_series, errors="coerce")
    wd_rad = np.deg2rad(wd)
    return np.sin(wd_rad) * w, np.cos(wd_rad) * w

# ==============================================================================
# Load station data
# ==============================================================================

renamed_stations = []
stations = {}

for name, path in station_files.items():
    df = pd.read_csv(path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = convert_to_numeric(df)

    if all(col in df.columns for col in required_columns):
        df_renamed = df[required_columns].copy()
        df_renamed = df_renamed.rename(columns={
            col: f"{col}_{name}" for col in required_columns if col != 'TIMESTAMP'
        })
        renamed_stations.append(df_renamed)
        stations[name] = df
    else:
        missing = [c for c in required_columns if c not in df.columns]
        print(f"WARNING: {name} missing columns: {missing}")

combined_df = reduce(
    lambda left, right: pd.merge(left, right, on='TIMESTAMP', how='outer'),
    renamed_stations
)
combined_df = combined_df.iloc[16:].reset_index(drop=True)
print(f"Combined shape after load: {combined_df.shape}")

# ==============================================================================
# Distance + IDW weighting
# FIX 1: Use .dropna().iloc[0] instead of hardcoded iloc[75164]
# ==============================================================================

def add_distance_to_target(combined_df, target_station):
    # FIX: was iloc[75164] — now finds first valid coordinate dynamically
    tgt_lat = combined_df[f'LAT_{target_station}'].dropna().iloc[0]
    tgt_lon = combined_df[f'LON_{target_station}'].dropna().iloc[0]

    for col in combined_df.columns:
        if col.startswith('LAT_'):
            name = col.replace('LAT_', '')
            if name != target_station and f'LON_{name}' in combined_df.columns:
                dist = haversine_distance(
                    combined_df[f'LAT_{name}'], combined_df[f'LON_{name}'],
                    tgt_lat, tgt_lon
                )
                combined_df[f'distance_to_{target_station}_{name}'] = dist
    return combined_df

def apply_idw_weights(combined_df, target_station, power=2):
    station_list = [s for s in station_files.keys() if s != target_station]
    base_vars = {
        col[: -(len(s) + 1)]
        for s in station_list
        for col in combined_df.columns
        if col.endswith(f'_{s}') and not col.startswith('LAT') and not col.startswith('LON')
    }

    for var in base_vars:
        for s in station_list:
            col_name = f"{var}_{s}"
            dist_col = f'distance_to_{target_station}_{s}'
            if col_name in combined_df.columns and dist_col in combined_df.columns:
                w = 1 / (combined_df[dist_col] ** power + 1e-6)
                combined_df[f'weighted_{var}_{s}'] = combined_df[col_name] * w

    return combined_df

# ==============================================================================
# Synoptic processing
# ==============================================================================

def process_synoptic_file(path):
    df = pd.read_csv(path, skiprows=[1])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True, errors='coerce').dt.tz_localize(None)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.round('15min')
    df = df[df['TIMESTAMP'].notna()]

    lat = df['LAT'].iloc[0] if 'LAT' in df.columns else None
    lon = df['LON'].iloc[0] if 'LON' in df.columns else None

    df = df[['TIMESTAMP', 'air_temp_set_1', 'relative_humidity_set_1',
             'wind_speed_set_1', 'wind_direction_set_1', 'Elevation']]

    df['wind_direction_set_1']    = pd.to_numeric(df['wind_direction_set_1'],    errors='coerce')
    df['wind_speed_set_1']        = pd.to_numeric(df['wind_speed_set_1'],        errors='coerce')
    df['air_temp_set_1']          = pd.to_numeric(df['air_temp_set_1'],          errors='coerce')
    df['relative_humidity_set_1'] = pd.to_numeric(df['relative_humidity_set_1'], errors='coerce')

    df   = df.groupby('TIMESTAMP', as_index=False).mean(numeric_only=True)
    temp = df.set_index('TIMESTAMP')

    if not isinstance(temp.index, pd.DatetimeIndex):
        raise TypeError("TIMESTAMP index is not DatetimeIndex after grouping.")

    # Circular wind direction interpolation
    wd_rad = np.deg2rad(temp['wind_direction_set_1'])
    u_i    = temp['wind_speed_set_1'] * np.sin(wd_rad)
    v_i    = temp['wind_speed_set_1'] * np.cos(wd_rad)

    try:
        u_interp  = u_i.resample('15min').interpolate('akima')
        v_interp  = v_i.resample('15min').interpolate('akima')
        wd_interp = np.rad2deg(np.arctan2(u_interp, v_interp))
        wd_interp = (wd_interp + 360) % 360
    except Exception:
        wd_interp = temp['wind_direction_set_1'].resample('15min').interpolate('nearest')

    df_int = temp[['air_temp_set_1', 'relative_humidity_set_1',
                   'wind_speed_set_1', 'Elevation']].resample('15min').interpolate('akima')

    out = df_int.reset_index()
    out['wind_direction_set_1'] = wd_interp.values
    out['LAT'] = lat
    out['LON'] = lon
    return out

def integrate_synoptic(df, combined_df_with_coords, target_station, synoptic_dfs):
    """
    FIX 2: was using stations[target_station]['LAT'].iloc[75164]
    Now correctly reads LAT_{target_station} from combined_df with first valid value.
    """
    start       = df['TIMESTAMP'].min()
    end         = df['TIMESTAMP'].max()
    station_lat = combined_df_with_coords[f'LAT_{target_station}'].dropna().iloc[0]
    station_lon = combined_df_with_coords[f'LON_{target_station}'].dropna().iloc[0]

    for i, syn_df in enumerate(synoptic_dfs):
        if syn_df.empty:
            continue
        syn_df  = syn_df[(syn_df['TIMESTAMP'] >= start) & (syn_df['TIMESTAMP'] <= end)].copy()
        syn_df  = convert_to_numeric(syn_df)
        merged  = pd.merge(df[['TIMESTAMP']], syn_df, on='TIMESTAMP', how='left')

        syn_lat = syn_df['LAT'].dropna().iloc[0]
        syn_lon = syn_df['LON'].dropna().iloc[0]
        dist    = max(haversine_distance(syn_lat, syn_lon, station_lat, station_lon), 0.1)
        weight  = 1 / (dist ** 2)

        sin_w, cos_w = circular_weight_components(merged['wind_direction_set_1'], weight)
        df[f'wind_direction_sin_weighted_{i}'] = sin_w.values
        df[f'wind_direction_cos_weighted_{i}'] = cos_w.values
        df[f'wind_speed_weighted_{i}']         = merged['wind_speed_set_1'].values        * weight
        df[f'air_temp_weighted_{i}']           = merged['air_temp_set_1'].values          * weight
        df[f'relative_humidity_weighted_{i}']  = merged['relative_humidity_set_1'].values * weight
        df[f'synoptic_elevation_{i}']          = syn_df['Elevation'].iloc[0]

    return df

# ==============================================================================
# FIX 3: drop only true intermediates, NOT weighted_* feature columns
# Old: 'weight_' in col matched 'weighted_SlrW_Avg_Afono' and dropped it
# New: col.startswith('weight_') only matches 'weight_Afono', 'weight_Poloa' etc.
# ==============================================================================

def drop_unused_columns(df):
    drop_cols = [
        col for col in df.columns
        if col.startswith('distance_')
        or col.startswith('weight_')    # raw IDW scalars only, NOT weighted_* features
        or col.startswith('LAT_')
        or col.startswith('LON_')
    ]
    print(f"Dropping intermediates: {drop_cols}")
    return df.drop(columns=drop_cols)

# ==============================================================================
# Run pipeline — CHANGE target_station as needed
# ==============================================================================

target_station = 'Vaipito'  # CHANGE THIS

synoptic_dfs = [process_synoptic_file(p) for p in synoptic_files]

combined_df = add_distance_to_target(combined_df, target_station)
combined_df = apply_idw_weights(combined_df, target_station)
combined_df = integrate_synoptic(combined_df, combined_df, target_station, synoptic_dfs)
combined_df = drop_unused_columns(combined_df)

print(f"Final shape: {combined_df.shape}")
print(f"Final columns:\n{combined_df.columns.tolist()}")

# ==============================================================================
# Optional: drop known missing sensor columns before splitting
# Uncomment as needed per target station/variable
# ==============================================================================
combined_df = combined_df.drop(columns=['SlrMJ_Tot_Poloa'],  errors='ignore')
combined_df = combined_df.drop(columns=['weighted_SlrMJ_Tot_Poloa'], errors='ignore')
combined_df = combined_df.drop(columns=['SlrW_Avg_Poloa'],             errors='ignore')
combined_df = combined_df.drop(columns=['SlrW_Avg_Vaipito'],           errors='ignore')
#combined_df = combined_df.drop(columns=['SlrMJ_Tot_Vaipito'],          errors='ignore')
combined_df = combined_df.drop(columns=['weighted_SlrW_Avg_Poloa'],    errors='ignore')
combined_df = combined_df.drop(columns=['weighted_SlrW_Avg_Vaipito'],  errors='ignore')
#combined_df = combined_df.drop(columns=['weighted_SlrMJ_Tot_Vaipito'], errors='ignore')
#combined_df = combined_df.drop(columns=['RH_Aasu'],          errors='ignore')
#combined_df = combined_df.drop(columns=['weighted_RH_Aasu'],          errors='ignore')

#combined_df = combined_df.drop(columns=['AirTF_Avg_Aasu'],   errors='ignore')
#combined_df = combined_df.drop(columns=['weighted_AirTF_Avg_Aasu'],   errors='ignore')

#combined_df = combined_df.drop(columns=['Rain_in_Tot_Aasu'], errors='ignore')
#combined_df = combined_df.drop(columns=['weighted_Rain_in_Tot_Aasu'], errors='ignore')


# ==============================================================================
# Date config
# ==============================================================================

date_config = {
    ('Vaipito', 'SlrW_Avg'): {
        "cutoff": "2022-03-23 10:30:00",
        "start":  "2022-03-23 10:30:00",
        "end":    "2022-08-14 13:30:00",
    },
    ('Vaipito', 'SlrMJ_Tot'): {
        "cutoff": "2022-03-23 10:30:00",
        "start":  "2022-03-23 10:30:00",
        "end":    "2022-08-14 13:30:00",
    },
    ('Aasu', 'RH'): {
        "cutoff": "2020-03-09 08:45:00",
        "start":  "2020-03-09 08:45:00",
        "end":    "2021-04-10 22:00:00",
    },
    ('Aasu', 'AirTF_Avg'): {
        "cutoff": "2020-03-09 08:45:00",
        "start":  "2020-03-09 08:45:00",
        "end":    "2021-04-10 22:00:00",
    },
    ('Poloa', 'SlrW_Avg'): {
        "cutoff": "2022-03-22 06:00:00",
        "start":  "2022-03-22 06:00:00",
        "end":    "2022-08-14 11:30:00",
    },
    ('Poloa', 'SlrMJ_Tot'): {
        "cutoff": "2022-03-22 06:00:00",
        "start":  "2022-03-22 06:00:00",
        "end":    "2022-08-14 11:30:00",
    },
    ('Aasu', 'Rain_in_Tot'): {
        "cutoff": "2020-04-14 00:20:00",
        "start":  "2020-04-14 00:20:00",
        "end":    "2022-03-25 13:45:00",
    },
}

# ==============================================================================
# Train / pred split
# ==============================================================================

def create_train_pred_splits(df, target_station, target_variable, config_dict,
                              simulate_missing=True):
    key = (target_station, target_variable)
    if key not in config_dict:
        raise ValueError(f"No date config found for key {key}")

    date_cfg = config_dict[key]
    cutoff   = pd.to_datetime(date_cfg['cutoff'])
    start    = pd.to_datetime(date_cfg['start'])
    end      = pd.to_datetime(date_cfg['end'])

    target_col          = f"{target_variable}_{target_station}"
    non_target_features = [c for c in df.columns if c not in ['TIMESTAMP', target_col]]

    df       = df.dropna(subset=non_target_features, how='all')
    df_train = df[df['TIMESTAMP'] <= cutoff].dropna()
    df_pred  = df[(df['TIMESTAMP'] > start) & (df['TIMESTAMP'] < end)].copy()

    if simulate_missing and target_col in df_pred.columns:
        df_pred[target_col] = np.nan

    print(f"Train: {df_train.shape}  {df_train['TIMESTAMP'].min()} → {df_train['TIMESTAMP'].max()}")
    print(f"Pred:  {df_pred.shape}   {df_pred['TIMESTAMP'].min()} → {df_pred['TIMESTAMP'].max()}")

    return df_train, df_pred

# ==============================================================================
# Example call — CHANGE as needed
# ==============================================================================

df_train, df_pred = create_train_pred_splits(
    combined_df,
    target_station='Vaipito',
    target_variable='SlrMJ_Tot',
    config_dict=date_config
)

df_train.to_csv("/Users/lizamclatchy/Documents/GitHub/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Model Input Data/vaipito_SlrMJ_Tot_train.csv", index=False)
# df_pred.to_csv(".../vaipito_SlrMJ_Tot_pred.csv", index=False)