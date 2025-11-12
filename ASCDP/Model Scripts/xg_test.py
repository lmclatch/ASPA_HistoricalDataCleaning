#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 07:58:12 2025

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
import optuna
from sklearn.model_selection import cross_val_score
#load cleaned_df from combined_df


#TEST FOR ONE VARIABLE.
combined_df = pd.read_csv("train_poloa_windspeed.csv")
#df_train = pd.read_csv("train_aasu_windspeed.csv")
#df_pred= pd.read_csv("pred_aasu_windspeed.csv")
#Filter non-NaN values for RH_Aasu
selected_columns = ['TIMESTAMP', 'WS_mph_S_WVT'] + [col for col in combined_df.columns if col not in ['TIMESTAMP', 'WS_mph_S_WVT']]
rh_data = combined_df[selected_columns].copy()
rh_data = rh_data.dropna(subset=['WS_mph_S_WVT'])  # Keep only rows where RH_Aasu is not NaN

    # Feature Engineering
def create_features(df):
        # Add more local wind features
    df['wind_speed_diff'] = df['WS_mph_S_WVT'].diff()
    df['HeatIndex_approx'] = df['AirTF_Avg'] * df['RH'] / 100
        # Add more direction-based features

    feature_cols = [col for col in df.columns if col not in ['TIMESTAMP']]
    
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag6'] = df[col].shift(6)
        df[f'{col}_rolling2'] = df[col].rolling(window=2).mean()
        df[f'{col}_rolling4'] = df[col].rolling(window=4).mean()
        df[f'{col}_rolling6'] = df[col].rolling(window=6).mean()
    
    # Add more granular time features
    df['hour_of_day'] = pd.to_datetime(df['TIMESTAMP']).dt.hour
    df['is_daytime'] = (df['hour_of_day'] >= 6) & (df['hour_of_day'] <= 18)

    return df

# Safe SMAPE calculation to handle zero values
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

# Updated Forecasting Function
def xgboost_forecast(train_df, target_column):
    # Sort by timestamp to maintain time series order
    train_df = train_df.sort_values(by="TIMESTAMP")
    # Train-Test Split BEFORE Feature Engineering
    train_data, test_data = train_test_split(train_df, test_size=0.2, shuffle=False)

    # Apply feature engineering separately to avoid leakage
    train_data = create_features(train_data)  # Apply AFTER split
    test_data = create_features(test_data)    # Apply AFTER split

    # Drop NaNs created by lag/rolling features
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Define features and target variable
    X_train = train_data.drop(columns=[target_column, 'TIMESTAMP'])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, 'TIMESTAMP'])
    y_test = test_data[target_column]

    # Visualize Train-Test Split
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(train_data.index, y_train, label='Training Set')
    ax.plot(test_data.index, y_test, label='Test Set', linestyle="dashed")
    ax.set_title(f'Data Train/Test Split for {target_column}')
    ax.legend()
    plt.show()

    # --- Optuna Bayesian Optimization instead of GridSearchCV ---
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'objective': 'reg:squarederror',
            'random_state': 42,
        }
        model = XGBRegressor(**params)
        # Use 3-fold cross-validation on training set
        scores = -1 * cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)  # You can increase n_trials for more thorough search

    print(f"Best Parameters: {study.best_params}")

    # Train best model on full training set
    best_model = XGBRegressor(**study.best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train, y_train)

    # Make Predictions with the Tuned Model
    y_pred = best_model.predict(X_test)



    # Evaluate with SMAPE
    error = smape(y_test, y_pred)
    print(f'SMAPE for {target_column}: {error:.4f}%')

    return best_model, y_test, y_pred


# Run forecast for Rain_in_Tot_Aasu
model, y_test, y_pred = xgboost_forecast(rh_data, 'WS_mph_S_WVT')



# Plot Actual vs Predicted RH_Aasu
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual WS_mph_S_WVT_Poloa', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted RWS_mph_S_WVT_Poloa', color='red', linestyle='--')
plt.xlabel('Index')
plt.ylabel('Speed (MPH)')
plt.title('Actual vs Predicted WS_mph_S_WVT_Poloa')
plt.legend()
plt.tight_layout()
plt.show()

from xgboost import plot_importance

# Plot feature importance
plot_importance(model, importance_type='weight',max_num_features=15, title = 'Feature Importance: Weight')  # Frequency of feature usage
plot_importance(model, importance_type='gain',max_num_features=10, title = 'Feature Importance: Gain')    # Contribution to performance
plot_importance(model, importance_type='cover',max_num_features=15, title = 'Feature Importance: Cover')   # Relative coverage in trees
plt.show()
   
residuals = y_test - y_pred

   # Plot residuals over time
plt.figure(figsize=(12, 5))
plt.plot(y_test.index, residuals, marker='o', linestyle='-', alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals Over Time')
plt.xlabel('Index')
plt.ylabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.show()
   # Plot residuals vs. predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs. Predicted')
plt.xlabel('Predicted Value')
plt.ylabel('Residual')
plt.tight_layout()
plt.show()
