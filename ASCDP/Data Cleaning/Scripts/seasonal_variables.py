#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:54:46 2024

@author: lizamclatchy
"""
import requests
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Load the dataset
aasu_df = pd.read_csv('aasu_seasonal_variables.csv')

# Ensure TIMESTAMP is properly formatted and set as the index
df_subset = aasu_df.copy()
try:
    df_subset['TIMESTAMP'] = pd.to_datetime(df_subset['TIMESTAMP'], errors='coerce')  # Ensure proper datetime format
    df_subset = df_subset.dropna(subset=['TIMESTAMP'])  # Drop rows where TIMESTAMP is NaT
    df_subset.set_index('TIMESTAMP', inplace=True)
except Exception as e:
    print(f"Error processing TIMESTAMP column: {e}")

# Ensure numeric columns and drop NaNs
try:
    for col in ['PTemp_C_Max', 'RH', 'SlrW_Avg', 'SlrMJ_Tot']:
        if col in df_subset.columns:
            df_subset[col] = pd.to_numeric(df_subset[col], errors='coerce')  # Convert to numeric
    df_subset = df_subset.dropna(subset=['PTemp_C_Max', 'RH', 'SlrW_Avg', 'SlrMJ_Tot'])  # Drop rows with NaN in required columns
except Exception as e:
    print(f"Error forcing numeric and dropping NaNs: {e}")

# Function to plot seasonal decomposition
def plot_solar_radiation(df, seasonal_period=96):
    """
    Plots seasonal decomposition of solar radiation variables (SlrW_Avg and SlrMJ_Tot).
    """
    # Check if required columns exist
    required_columns = ['SlrW_Avg', 'SlrMJ_Tot']
    for column in required_columns:
        if column not in df.columns:
            print(f"Missing required column: {column}")
            return
    
    # Drop rows where both SlrW_Avg and SlrMJ_Tot are NaN
    df = df.dropna(subset=['SlrW_Avg', 'SlrMJ_Tot'], how='all')
    
    # Initialize subplots
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
    fig.suptitle("Seasonal Decomposition of Solar Radiation Variables", fontsize=16)

    # Decompose SlrW_Avg
    if not df['SlrW_Avg'].dropna().empty:  # Ensure non-empty data
        try:
            result_slrw_avg = seasonal_decompose(df['SlrW_Avg'].dropna(), model='additive', period=seasonal_period)
            result_slrw_avg.observed.plot(ax=axs[0, 0], color='orange', title='Observed SlrW_Avg')
            axs[0, 0].set_ylabel('Observed')

            result_slrw_avg.trend.plot(ax=axs[1, 0], color='orange', title='Trend in SlrW_Avg')
            axs[1, 0].set_ylabel('Trend')

            result_slrw_avg.seasonal.plot(ax=axs[2, 0], color='orange', title='Seasonal SlrW_Avg')
            axs[2, 0].set_ylabel('Seasonal')

            result_slrw_avg.resid.plot(ax=axs[3, 0], color='orange', title='Residuals of SlrW_Avg')
            axs[3, 0].set_ylabel('Residual')
            axs[3, 0].set_xlabel('Date')
        except Exception as e:
            print(f"Error decomposing SlrW_Avg: {e}")
    else:
        print("No valid data for SlrW_Avg.")

    # Decompose SlrMJ_Tot
    if not df['SlrMJ_Tot'].dropna().empty:  # Ensure non-empty data
        try:
            result_slrmj_tot = seasonal_decompose(df['SlrMJ_Tot'].dropna(), model='additive', period=seasonal_period)
            result_slrmj_tot.observed.plot(ax=axs[0, 1], color='green', title='Observed SlrMJ_Tot')
            axs[0, 1].set_ylabel('Observed')

            result_slrmj_tot.trend.plot(ax=axs[1, 1], color='green', title='Trend in SlrMJ_Tot')
            axs[1, 1].set_ylabel('Trend')

            result_slrmj_tot.seasonal.plot(ax=axs[2, 1], color='green', title='Seasonal SlrMJ_Tot')
            axs[2, 1].set_ylabel('Seasonal')

            result_slrmj_tot.resid.plot(ax=axs[3, 1], color='green', title='Residuals of SlrMJ_Tot')
            axs[3, 1].set_ylabel('Residual')
            axs[3, 1].set_xlabel('Date')
        except Exception as e:
            print(f"Error decomposing SlrMJ_Tot: {e}")
    else:
        print("No valid data for SlrMJ_Tot.")

    # Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Plot seasonal decomposition for the cleaned DataFrame
plot_solar_radiation(df_subset, seasonal_period=96)

        
# p = d = q = range(0, 3)  
# pdq = list(itertools.product(p, d, q))  
# def optimize_arima(series, pdq):
#     best_aic = np.inf
#     best_params = None
#     best_fit = None
    
#     for param in pdq:
#         try:
#             model = SARIMAX(series,
#                             order=param,
#                             enforce_stationarity=False,
#                             enforce_invertibility=False)
#             results = model.fit(disp=False)
#             if results.aic < best_aic:
#                 best_aic = results.aic
#                 best_params = param
#                 best_fit = results
#         except Exception as e:
#             continue
            
#     return best_params, best_aic, best_fit

# #Optimize ARIMA for Precipitation Data
# best_slw_params, best_slw_aic, best_slw_fit = optimize_arima(df_subset['SlrW_Avg'], pdq)
# print(f'Best ARIMA parameters for SlrW_Avg: {best_slw_params} with AIC: {best_slw_aic}')

# #Optimize ARIMA for Discharge Data
# best_slrmj_tot_params, best_slrmj_tot_aic, best_slrmj_tot_fit = optimize_arima(df_subset['SlrMJ_Tot'], pdq)
# print(f'Best ARIMA parameters for SlrMJ_Tot data: {best_slrmj_tot_params} with AIC: {best_slrmj_tot_aic}')


