#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 21 09:23:14 2025

@author: lizamclatchy
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from functools import reduce

'''
Loading all relevant .csvs, other weather stations and synoptic data

Note: Target variable is the variable you are trying to forecast
Target station is the station of the variable you want to forecast
'''
aasu_df = pd.read_csv('Aasu/Aasu_ALL_15min_data_cleaned.csv')
poloa_df = pd.read_csv('Poloa/Poloa_ALL_15min_data_cleaned.csv')
afono_df = pd.read_csv('Afono/Afono_ALL_15min_data_cleaned.csv')
vaipito_df = pd.read_csv('Vaipito/Vaipito_ALL_15min_data_cleaned.csv')
cols_to_drop = ['WS_mph_S_WVT', 'WindDir_D1_WVT', 'WindDir_SD1_WVT', 'RECORD', 'BattV_Avg']

for df in [aasu_df,vaipito_df,poloa_df,afono_df]:
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

stations = {
    'Aasu': aasu_df,
    'Poloa': poloa_df,
    'Vaipito': vaipito_df,
    'Afono': afono_df,
}

''' 
Prepping station data
Functions below are ensuring location data is being considered 
'''

# --- Haversine Distance Function ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # Distance in kilometers

# --- Function to Add Distance to Target Station ---
def add_distance_to_target(combined_df, target_station):
    # Extract target station's latitude & longitude
    target_lat = combined_df[f'LAT_{target_station}'].iloc[75164]
    target_lon = combined_df[f'LON_{target_station}'].iloc[75164]

    # Calculate distance for each station to the target station
    for col in combined_df.columns:
        if 'LAT_' in col:
            station_name = col.split('_')[1]  #Extract station name
            if station_name != target_station:
                lat_col = f'LAT_{station_name}'
                lon_col = f'LON_{station_name}'
                combined_df[f'distance_to_{target_station}_{station_name}'] = haversine_distance(
                    combined_df[lat_col], combined_df[lon_col],
                    target_lat, target_lon
                )
    return combined_df
# Apply IDW to Rainfall Data from Other Stations
def apply_idw_weights(combined_df, target_station='Aasu', power=2): #Change target station
    """
    Applies IDW to all numerical variables dynamically for all stations except the target.
    """
    station_list = [s for s in ['Poloa', 'Vaipito', 'Afono'] if s != target_station] #change for target station

    # Identify base variable names (e.g., 'Rain_in_Tot') without station suffixes
    base_vars = set()
    for col in combined_df.columns:
        for station in station_list:
            suffix = f"_{station}"
            if col.endswith(suffix):
                base_vars.add(col.replace(suffix, ""))

    # Compute weights and weighted values
    for var in base_vars:
        for station in station_list:
            col_name = f"{var}_{station}"
            if col_name in combined_df.columns:
                dist_col = f'distance_to_{target_station}_{station}'
                weight_col = f'weight_{station}'

                # Ensure no divide-by-zero
                combined_df[weight_col] = 1 / (combined_df[dist_col] ** power + 1e-6)

                # Create weighted column
                combined_df[f'weighted_{var}_{station}'] = combined_df[col_name] * combined_df[weight_col]

    return combined_df
# Define required columns / can change based on target variable
required_columns = ['TIMESTAMP','LAT','LON','PTemp_C_Max','AirTF_Avg','SlrW_Avg','SlrMJ_Tot']
#Filter and rename columns for each station
renamed_stations = []
for name, df in stations.items():
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {name}: Missing required columns.")
        continue
    
    # Rename columns to include station name
    df_renamed = df[required_columns].copy()
    df_renamed = df_renamed.rename(columns={col: f"{col}_{name}" for col in required_columns if col != 'TIMESTAMP'})
    renamed_stations.append(df_renamed)
    
'''
All stations have been prepped are being merged. 
Removing the first 16 indcies because they don't match for all the stations
Removing non-target wind data, making the dataframe numeric/still prepping only station data 
'''

combined_df = reduce(lambda left, right: pd.merge(left, right, on='TIMESTAMP', how='outer'), renamed_stations)
combined_df.drop(combined_df.index[:16], axis=0, inplace=True)
# SPECIFIC FOR WIND DATA

def convert_to_numeric(df):
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
combined_df = convert_to_numeric(combined_df)
combined_df = add_distance_to_target(combined_df, target_station='Aasu')
combined_df = apply_idw_weights(combined_df, power=2)
combined_df = combined_df.loc[:100696] 

target_lat = aasu_df["LAT"].iloc[75164]  #Change to target station
target_lon = aasu_df["LON"].iloc[75164]
'''
Editing the combined_df, removing non useful columns, this needs to be changed between wind and rainfall
Change this based on your varying target station and target variable
Wind is tricky because so much is missing
'''

non_target_features = [col for col in combined_df.columns if col != 'RH_Aasu']
combined_df = combined_df.dropna(subset=non_target_features, how='all')
def drop_unused_columns(combined_df):
    # Drop all LAT, LON, and weight columns
    drop_cols = [col for col in combined_df.columns if 'distance' in col or 'weight_' in col or 'LON' in col or 'LAT' in col]
    combined_df = combined_df.drop(columns=drop_cols)
    return combined_df
combined_df = drop_unused_columns(combined_df)

#Export file
combined_df = convert_to_numeric(combined_df)
combined_df = combined_df.dropna()
cutoff_date = pd.to_datetime("2020-03-09 08:45:00")
combined_df = combined_df[combined_df["TIMESTAMP"] <= cutoff_date]
combined_df=combined_df.drop(columns=['AirTF_Avg_Aasu'])
combined_df.to_csv("Aasu/aasu_rh_train.csv",index=False)

