#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 10:30:18 2025

@author: lizamclatchy
"""
import pandas as pd
import numpy as np

def convert_to_numeric(df):
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- Haversine Distance Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    

    return R * c  # Distance in kilometers

def remove_outliers_iqr(df, multiplier=3.0):
    """
    Remove outliers from all numeric columns using the IQR method.
    Keeps TIMESTAMP and non-numeric columns unchanged.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        # Replace extreme values with NaN (so they get handled naturally in interpolation)
        df_clean[col] = df_clean[col].mask((df_clean[col] < lower) | (df_clean[col] > upper))
    
    return df_clean

def process_synoptic_file(file_path: str, skip_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=[1])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True, errors='coerce').dt.tz_localize(None)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.round('15min')
    df = df[df['TIMESTAMP'].notna()]

    lat = df['LAT'].iloc[0] if 'LAT' in df.columns else None
    lon = df['LON'].iloc[0] if 'LON' in df.columns else None

    df = df[['TIMESTAMP','air_temp_set_1','relative_humidity_set_1',
             'wind_speed_set_1','wind_direction_set_1','Elevation']]

    df['wind_direction_set_1']   = pd.to_numeric(df['wind_direction_set_1'], errors='coerce')
    df['wind_speed_set_1']       = pd.to_numeric(df['wind_speed_set_1'], errors='coerce')
    df['air_temp_set_1']         = pd.to_numeric(df['air_temp_set_1'], errors='coerce')
    df['relative_humidity_set_1']= pd.to_numeric(df['relative_humidity_set_1'], errors='coerce')
    df = df.groupby('TIMESTAMP', as_index=False).mean(numeric_only=True)

    try:
        df_resampled = (df.set_index('TIMESTAMP').resample('15min').interpolate(method='akima').reset_index())
    except Exception:
        df_resampled = (df.set_index('TIMESTAMP').resample('15min').interpolate(method='time').reset_index())

    if skip_rows:
        df_resampled = df_resampled.iloc[skip_rows:].reset_index(drop=True)

    # Always reattach synoptic station coords
    df_resampled['LAT'] = lat
    df_resampled['LON'] = lon
    return df_resampled


synoptic_resample_df = process_synoptic_file('/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Raw Data/NSTU.2022-12-31.csv') #, skip_rows=2249)
synoptic_resample_df_1 = process_synoptic_file('/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Raw Data/SFGP6.2022-12-31.csv') #, skip_rows=2248)


def run_forecast_pipeline(
    station_df,
    station_name,
    target_column="WS_mph_S_WVT",
    synoptic_dfs=[],
    feature_columns=["PTemp_C_Max","AirTF_Avg","RH","SlrW_Avg","SlrMJ_Tot","Elevation"],#,"Rain_in_Tot"],
    dropna_thresh=0.5,
    station_lat: float = None,     # <<< required for IDW
    station_lon: float = None,     # <<< required for IDW
    normalize_weights: bool = False # <<< make weights sum to 1
):
    if station_lat is None or station_lon is None:
        raise ValueError("station_lat and station_lon must be provided for distance-based weighting.")
    
    station_df = station_df.copy()
    station_df["TIMESTAMP"] = pd.to_datetime(station_df["TIMESTAMP"])
    station_df = convert_to_numeric(station_df)
    station_df = remove_outliers_iqr(station_df)
    # FIX: target + features selection
    required_columns = ["TIMESTAMP", target_column] + feature_columns
    station_df = station_df[[c for c in required_columns if c in station_df.columns]]

    # Rename target-side features
    station_df = station_df.rename(columns={
        'RH':'RH_target','AirTF_Avg':'AirTF_target','Rain_in_Tot':'Rain_target',
        'PTemp_C_Max':'PTemp_target','SlrW_Avg':'SolarW_target','SlrMJ_Tot':'SolarMJ_target',
        'Elevation':'Elevation_target'
    })

    # Time window for merges
    start = station_df['TIMESTAMP'].min()
    end   = station_df['TIMESTAMP'].max()

    # ---- compute per-synoptic weights via IDW (1/d^2)
    weights = []
    valid_syn = []
    for syn_df in synoptic_dfs:
        if syn_df is None or syn_df.empty or 'LAT' not in syn_df.columns or 'LON' not in syn_df.columns:
            weights.append(0.0)
            valid_syn.append(None)
            continue
        # Use the synoptic station’s fixed coords (first row is fine)
        syn_lat = syn_df['LAT'].iloc[0]
        syn_lon = syn_df['LON'].iloc[0]
        d = max(haversine_distance(syn_lat, syn_lon, station_lat, station_lon), 0.1)  # avoid 0
        w = 1.0 / (d**2)
        weights.append(w)
        valid_syn.append(syn_df)

    # Normalize if requested (keeps magnitudes stable across #stations)
    wsum = sum(weights)
    if normalize_weights and wsum > 0:
        weights = [w/wsum for w in weights]

    # ---- merge each synoptic stream and apply its weight
    for i, (syn_df, w) in enumerate(zip(valid_syn, weights)):
        if syn_df is None:
            # still create the columns (as NaN) to keep schema consistent
            station_df[f'wind_direction_weighted_{i}']    = np.nan
            station_df[f'wind_speed_weighted_{i}']        = np.nan
            station_df[f'air_temp_weighted_{i}']          = np.nan
            station_df[f'relative_humidity_weighted_{i}'] = np.nan
            continue

        # Align to station time window and select columns
        s = syn_df[(syn_df['TIMESTAMP'] >= start) & (syn_df['TIMESTAMP'] <= end)][
            ['TIMESTAMP','wind_direction_set_1','wind_speed_set_1',
             'air_temp_set_1','relative_humidity_set_1','Elevation']
        ].copy()

        merged = station_df[['TIMESTAMP']].merge(
            s, on='TIMESTAMP', how='left'
        )

        # Apply weight (IDW)
        station_df[f'wind_direction_weighted_{i}']    = merged['wind_direction_set_1']     * w
        station_df[f'wind_speed_weighted_{i}']        = merged['wind_speed_set_1']         * w
        station_df[f'air_temp_weighted_{i}']          = merged['air_temp_set_1']           * w
        station_df[f'relative_humidity_weighted_{i}'] = merged['relative_humidity_set_1']  * w

    # (Optional) you can also add combined “weighted-sum” features:
    # station_df['wind_speed_weighted_sum'] = station_df.filter(like='wind_speed_weighted_').sum(axis=1)

    # Train/pred split (keep your station-specific cutoff logic)
    df_train = station_df[station_df[target_column].notna()]
    cutoffs = {
        "Aasu":   "2019-08-27 20:30",
        "Afono":  "2019-08-16 14:00",
        "Vaipito":"2020-10-14 20:00",
        "Poloa":  "2020-06-20 05:15",
    }
    cutoff_date = pd.to_datetime(cutoffs.get(station_name, "2100-01-01 00:00"))

    df_train = df_train[df_train["TIMESTAMP"] <= cutoff_date]
    df_pred  = station_df[(station_df[target_column].isna()) & (station_df["TIMESTAMP"] > cutoff_date)]

    return df_train, df_pred


    return df_train, df_pred


#CHANGE THIS BASED ON STATION AND VARIABLE
df_train, df_pred = run_forecast_pipeline(
    station_df=pd.read_csv('/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Raw Data/Poloa_ALL_15min_data_cleaned.csv'),
    station_name="Poloa",
    target_column="WS_mph_S_WVT",
    synoptic_dfs=[synoptic_resample_df, synoptic_resample_df_1],
    station_lat=-14.282,   # <<< your station’s lat, change based on station
    station_lon=-170.710,  # <<< your station’s lon, change based on station
    normalize_weights=False
    )
#CHANGE THIS NAME
df_train.to_csv("/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Model Input Data/train_poloa_WS_mph_S_WVT.csv", index=False)
#df_pred.to_csv("/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Model Input Data/pred_aasu_WindDir_D1_WVT.csv", index=False)
