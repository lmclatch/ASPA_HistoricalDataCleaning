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

def process_synoptic_file(file_path: str, skip_rows: int = 0) -> pd.DataFrame:
    df = pd.read_csv(file_path, skiprows=[1])
    
    # Parse and clean timestamp
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True, errors='coerce').dt.tz_localize(None)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.round('15min')
    df = df[df['TIMESTAMP'].notna()]

    print(df.columns)

    # Get lat/lon (before dropping!)
    lat = df['LAT'].iloc[0] if 'LAT' in df.columns else None
    lon = df['LON'].iloc[0] if 'LON' in df.columns else None

    # Keep only columns we care about
    df = df[['TIMESTAMP', 'air_temp_set_1', 'relative_humidity_set_1',
             'wind_speed_set_1', 'wind_direction_set_1','Elevation']]

    # Coerce numeric
    df['wind_direction_set_1'] = pd.to_numeric(df['wind_direction_set_1'], errors='coerce')
    df['wind_speed_set_1'] = pd.to_numeric(df['wind_speed_set_1'], errors='coerce')
    df['air_temp_set_1'] = pd.to_numeric(df['air_temp_set_1'], errors='coerce')
    df['relative_humidity_set_1'] = pd.to_numeric(df['relative_humidity_set_1'], errors='coerce')

    # Group by time (remove duplicate timestamps)
    df = df.groupby('TIMESTAMP', as_index=False).mean(numeric_only=True)

    # Resample to regular time intervals
    df_resampled = (
        df.set_index('TIMESTAMP')
          .resample('15min')
          .interpolate(method='akima')
          .reset_index()
    )

    # Skip initial rows if needed
    if skip_rows:
        df_resampled = df_resampled.iloc[skip_rows:].reset_index(drop=True)

    # Re-attach LAT/LON consistently
    df_resampled['LAT'] = lat
    df_resampled['LON'] = lon

    return df_resampled

synoptic_resample_df = process_synoptic_file('/Users/lizamclatchy/ASCDP/NSTU.2022-12-31.csv') #, skip_rows=2249)
synoptic_resample_df_1 = process_synoptic_file('/Users/lizamclatchy/ASCDP/SFGP6.2022-12-31.csv') #, skip_rows=2248)

def run_forecast_pipeline(
    station_df,
    station_name,
    target_column="WS_mph_S_WVT",
    synoptic_dfs=[],
    feature_columns=[
        "PTemp_C_Max", "AirTF_Avg", "RH", "Rain_in_Tot", "SlrW_Avg", "SlrMJ_Tot",'Elevation'
    ],
    dropna_thresh=0.5,
    skip_rows_synoptic=0,
    output_path=None
):
    station_df = station_df.copy()
    station_df["TIMESTAMP"] = pd.to_datetime(station_df["TIMESTAMP"])
    station_df = convert_to_numeric(station_df)

    # Use all available required columns
    required_columns = ["TIMESTAMP", target_column] + feature_columns
    available_columns = [col for col in required_columns if col in station_df.columns]
    station_df = station_df[available_columns]

    start = station_df['TIMESTAMP'].min()
    end = station_df['TIMESTAMP'].max()
    
    target_rename_map = {
        'RH': 'RH_target',
        'AirTF_Avg': 'AirTF_target',
        'Rain_in_Tot': 'Rain_target',
        'PTemp_C_Max': 'PTemp_target',
        'SlrW_Avg': 'SolarW_target',
        'SlrMJ_Tot': 'SolarMJ_target',
        'Elevation':'Elevation_target'
        #'WS_mph_S_WVT': 'WS_target',
     
    }

    station_df = station_df.rename(columns=target_rename_map)

    
    # Loop through synoptic stations
    for i, syn_df in enumerate(synoptic_dfs):
        if not syn_df.empty:
            syn_df = syn_df[(syn_df['TIMESTAMP'] >= start) & (syn_df['TIMESTAMP'] <= end)].copy()
            syn_df = syn_df[['TIMESTAMP', 'wind_direction_set_1', 'wind_speed_set_1', 'LAT', 'LON', 'air_temp_set_1', 'relative_humidity_set_1', 'Elevation']]

            merged = pd.merge(
                station_df[['TIMESTAMP']],
                syn_df[['TIMESTAMP', 'wind_direction_set_1', 'wind_speed_set_1','air_temp_set_1','relative_humidity_set_1','Elevation']],
                on='TIMESTAMP',
                how='left'
            )

            syn_lat = syn_df['LAT'].iloc[0]
            syn_elevation = syn_df['Elevation'].iloc[0]
            syn_lon = syn_df['LON'].iloc[0]
            tgt_lat = station_df['LAT'].iloc[0] if 'LAT' in station_df.columns else 0
            tgt_lon = station_df['LON'].iloc[0] if 'LON' in station_df.columns else 0

            # Compute distance and weight
            distance = max(haversine_distance(syn_lat, syn_lon, tgt_lat, tgt_lon), 0.1)
            weight = 1 / (distance ** 2)

            # Weighted wind inputs
            station_df[f'wind_direction_weighted_{i}'] = merged['wind_direction_set_1'] * weight
            station_df[f'wind_speed_weighted_{i}'] = merged['wind_speed_set_1'] * weight
            station_df[f'air_temp_weighted_{i}'] = merged['wind_speed_set_1'] * weight
            station_df[f'relative_humidity_weighted_{i}'] = merged['wind_speed_set_1'] * weight
            
            
            # Add explicit synoptic lat/lon/dist as features
            station_df[f'synoptic_elevation_{i}'] = syn_elevation
            #station_df[f'synoptic_LON_{i}'] = syn_lon
            #station_df[f'synoptic_distance_{i}'] = distance

    # Drop rows with too much missing data
    station_df = station_df.dropna(thresh=int(len(station_df.columns) * dropna_thresh))

    # Log-transform target (optional)
    if target_column in station_df.columns:
        station_df[target_column] = np.log1p(station_df[target_column])

    # Split into training and prediction datasets
    df_train = station_df[station_df[target_column].notna()]
    cutoff_date = pd.to_datetime("2019-08-27 20:30:00") 
    df_train = df_train[df_train["TIMESTAMP"] <= cutoff_date]

    df_pred = station_df[
        (station_df[target_column].isna()) &
        (station_df["TIMESTAMP"] > cutoff_date)
    ]

    return df_train, df_pred

df_train, df_pred = run_forecast_pipeline(
    station_df=pd.read_csv('Aasu_ALL_15min_data_cleaned.csv'),
    station_name="Aasu",
    target_column="WS_mph_S_WVT",  # <- use the column name that exists in your file
    synoptic_dfs=[synoptic_resample_df, synoptic_resample_df_1],
)

df_train.to_csv("train_aasu_windspeed.csv", index=False)
df_pred.to_csv("pred_aasu_windspeed.csv", index=False)
