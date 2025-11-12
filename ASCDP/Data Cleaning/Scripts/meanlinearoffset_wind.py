#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 11:43:07 2025

@author: lizamclatchy
"""

import pandas as pd
import numpy as np

station_df = pd.read_csv(
    '/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Raw Data/Poloa_ALL_15min_data_cleaned.csv',
    parse_dates=['TIMESTAMP'])
station_df = station_df[['TIMESTAMP', 'WS_mph_S_WVT']]

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

synoptic_resample_df = process_synoptic_file('/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Raw Data/NSTU.2022-12-31.csv') #, skip_rows=2249)
def convert_to_numeric(df):
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
station_df = convert_to_numeric(station_df)

# --- Merge on timestamp ---
merged_df = station_df.merge(
    synoptic_resample_df[['TIMESTAMP', 'wind_speed_set_1']],
    on='TIMESTAMP',
    how='left'
)

# --- Compute mean offset from overlapping values ---
valid = merged_df.dropna(subset=['WS_mph_S_WVT', 'wind_speed_set_1']).copy()
offset = (valid['WS_mph_S_WVT'] - valid['wind_speed_set_1']).mean()


# --- Fill missing station wind speeds using offset ---
pred_mask = merged_df['WS_mph_S_WVT'].isna() & merged_df['wind_speed_set_1'].notna()
merged_df.loc[pred_mask, 'WS_mph_S_WVT_filled'] = merged_df.loc[pred_mask, 'wind_speed_set_1'] + offset

# --- Predict for training rows to evaluate error ---
merged_df.loc[valid.index, 'WS_mph_S_WVT_pred'] = valid['wind_speed_set_1'] + offset
errors = merged_df.loc[valid.index, 'WS_mph_S_WVT_pred'] - valid['WS_mph_S_WVT']

mae = np.mean(np.abs(errors))
rmse = np.sqrt(np.mean(errors ** 2))
print(f"Mean wind speed offset = {offset:.2f} mph")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(merged_df['TIMESTAMP'], merged_df['WS_mph_S_WVT'], label='Observed (Poloa)', color='black', linewidth=1)
plt.plot(merged_df.loc[valid.index, 'TIMESTAMP'],
         merged_df.loc[valid.index, 'WS_mph_S_WVT_pred'],
         label='Predicted (Train)', linestyle='dotted', color='green', alpha=0.6)

plt.xlabel('Timestamp')
plt.ylabel('Wind Speed (mph)')
plt.title('Poloa Wind Speed: Observed vs. Offset-Filled (NSTU)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print(f"Training MAE: {mae:.2f} mph")
print(f"Training RMSE: {rmse:.2f} mph")