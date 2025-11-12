#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:58:33 2024

@author: lizamclatchy
"""

import requests
import numpy as np
import pandas as pd
from io import StringIO
import re
import matplotlib.pyplot as plt
import skgstat as skg
from skgstat import Variogram


aasu_spatial = pd.read_csv('Aasu_ALL_15min_data_spatial_variables.csv')
poloa_spatial = pd.read_csv('Poloa_ALL_15min_data_spatial_variables.csv')
vaipito_spatial = pd.read_csv('Vaipito_ALL_15min_data_spatial_variables.csv')
afono_spatial = pd.read_csv('Afono_ALL_15min_data_spatial_variables.csv')

stations = {
    'Aasu': aasu_spatial,
    'Poloa': poloa_spatial,
    'Vaipito': vaipito_spatial,
    'Afono': afono_spatial
}
poloa_spatial.drop(columns=['BattV_Avg.1'],inplace=True)
for name, df in stations.items():
    required_columns = ['TIMESTAMP', 'LAT', 'LON', 'Rain_in_Tot', 'PTemp_C_Max',
                        'AirTF_Avg','RH','Rain_in_Tot',
                        'WS_mph_S_WVT','WindDir_D1_WVT','WindDir_SD1_WVT']
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {name}: Missing required columns.")
        continue
    stations[name] = df.dropna(subset=required_columns)

def standardize_timestamps(stations):
    standardized_stations = {}
    for name, df in stations.items():
        print(f"Processing {name}...")
        
        # Drop rows with invalid headers or non-numeric data
        if name == 'Afono' and 'TS' in df.iloc[0].values:
            df = df.iloc[1:]  # Skip the first row for Afono
        
        # Attempt to convert TIMESTAMP column to datetime
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce', infer_datetime_format=True)
        
        # Drop rows where TIMESTAMP could not be converted
        df = df.dropna(subset=['TIMESTAMP'])
        
        # Sort data by TIMESTAMP for consistency
        df = df.sort_values('TIMESTAMP')
        
        # Store the cleaned dataframe
        standardized_stations[name] = df
        print(f"{name} has {len(df)} valid rows after cleaning.")
    
    return standardized_stations

# Standardize timestamps for all stations
standardized_stations = standardize_timestamps(stations)

# Verify the outputs
for name, df in standardized_stations.items():
    print(f"\n{name} - First 5 Rows:")
    print(df.head())

#Doing PTemp_C_Max as example for Afono and Aasu
#Step 1 pre process data
def preprocess_data(aasu_spatial, afono_spatial, value_column='PTemp_C_Max'):
    """
    Preprocesses data for time-series forecasting
    - Aligns the two datasets based on matching timestamps.
    - Returns aligned DataFrame with coordinates and values.
    """
    # Merge the data on TIMESTAMP
    merged_df = pd.merge(aasu_spatial, afono_spatial, on='TIMESTAMP', suffixes=('_1', '_2'))
   
    non_number_columns = ['TIMESTAMP']
    number_columns = [col for col in merged_df.columns if col not in non_number_columns]
    merged_df = merged_df.dropna()
    merged_df[number_columns] = merged_df[number_columns].apply(pd.to_numeric, errors='coerce')

    # Filter rows to have no missing data 
    
    # Debugging: Print number of aligned timestamps
    print(f"Number of matched timestamps in known data: {len(merged_df)}")

    return merged_df
combined_aasu_afono = preprocess_data(aasu_spatial, afono_spatial)





                
