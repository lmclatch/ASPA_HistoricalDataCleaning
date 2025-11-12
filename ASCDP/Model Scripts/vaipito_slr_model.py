#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 07:14:31 2025

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
from sklearn.metrics import mean_absolute_error,r2_score

#Using a regression model to adequately predict wind data, not forecasting, learning and predicting missing
combined_df = pd.read_csv("/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Model Input Data/vaipito_SlrMJ_Tot_train.csv")
selected_columns = ['TIMESTAMP', 'SlrMJ_Tot_Vaipito'] + [col for col in combined_df.columns if col not in ['TIMESTAMP', 'SlrMJ_Tot_Vaipito']]
rh_data = combined_df[selected_columns].copy()
rh_data = rh_data.dropna(subset=['SlrMJ_Tot_Vaipito'])  # Keep only rows where RH_Aasu is not NaN


def feature_engineering(df):
    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    target_column = 'SlrMJ_Tot_Vaipito'
    feature_cols = [col for col in df.columns if col not in ['TIMESTAMP', target_column,'Elevation_target','synoptic_elevation_1','synoptic_elevation_0']]    
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(2)
        df[f'{col}_lag3'] = df[col].shift(5)
        df[f'{col}_lag6'] = df[col].shift(7)
        df[f'{col}_rolling2'] = df[col].rolling(window=2).mean()
        df[f'{col}_rolling4'] = df[col].rolling(window=4).mean()
        df[f'{col}_rolling6'] = df[col].rolling(window=6).mean()
    
    # Add more granular time features
    df['hour_of_day'] = pd.to_datetime(df['TIMESTAMP']).dt.hour
    df['is_daytime'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] <= 18)).astype(int)
    # Day of Week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['TIMESTAMP'].dt.dayofweek
    # Season (simplified meteorological)
    df['month'] = df['TIMESTAMP'].dt.month
    def get_season(month):
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    df['season'] = df['month'].apply(get_season)
    # Enforce season as categorical with all 4 categories
    df['season'] = pd.Categorical(df['season'], categories=["winter", "spring", "summer", "fall"])
    # One-hot encode with fixed categories
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    season_dummies = season_dummies.astype(int)
    df = pd.concat([df, season_dummies], axis=1)
    df.drop(columns=['season'], inplace=True)

    
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
# Safe SMAPE calculation to handle zero values
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

def xgboost_regression(train_df, target_column):
    # Sort by timestamp to maintain time series order
    train_df = train_df.sort_values(by="TIMESTAMP")
    # Train-Test Split BEFORE Feature Engineering
    train_data, test_data = train_test_split(train_df, test_size=0.2, shuffle=False)

    # Apply feature engineering separately to avoid leakage
    train_data = feature_engineering(train_data)  # Apply AFTER split
    test_data = feature_engineering(test_data)    # Apply AFTER split

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
    # Bayesian Hyperparameter Tuning with Optuna
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return smape(y_test.values, preds)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best Params:", study.best_params)

    # Fit Best Model
    best_model = XGBRegressor(**study.best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train, y_train)
    # from sklearn.linear_model import LinearRegression
    # lr = LinearRegression().fit(X_train, y_train)
    # print("MAE (Linear):", mean_absolute_error(y_test, lr.predict(X_test)))
    y_pred = best_model.predict(X_test)

    # Evaluate SMAPE
    # smape_error = smape(y_test, y_pred)
    # print(f"SMAPE: {smape_error:.2f}%")
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.4f} W/m^2")
    r2 = r2_score(y_test, y_pred)
    print(f"R²:    {r2:.4f}")
   

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--', color='red')
    plt.title(f'Actual vs Predicted SlrMJ_Tot_Vaipito{target_column})')
    plt.xlabel("Index")
    plt.ylabel("")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_model, y_test, y_pred, X_train.columns.tolist()


model, y_test, y_pred, X_model_cols = xgboost_regression(rh_data, 'SlrMJ_Tot_Vaipito')
fe_df = feature_engineering(rh_data)
#fe_df.set_index("TIMESTAMP").to_csv("feature_engineered_data_with_index.csv")

plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual SlrMJ_Tot_Vaipito', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted SllrMJ_Tot_Vaipito', color='red', linestyle='--')
plt.xlabel('Index')
plt.ylabel('')
plt.title('Actual vs Predicted SlrW_Avg_Vaipito')
plt.legend()
plt.tight_layout()
plt.show()

from xgboost import plot_importance
plot_importance(model, importance_type='gain', max_num_features=15, title='XGBoost Feature Importance')
plt.tight_layout()
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(12, 4))
plt.plot(residuals, label='Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.title("Prediction Residuals Over Time")
plt.legend()
plt.tight_layout()
plt.show()

# #####apply  to df_pred fofr RH-Aasu

# # Load prediction data
# df_pred_raw = pd.read_csv("aasu_rh_pred.csv")

# # Apply same feature engineering
# df_pred_fe = feature_engineering(df_pred_raw)
# # Recreate the training-engineered DataFrame to align columns
# X_train_fe = feature_engineering(rh_data)
# X_model_cols = [col for col in X_train_fe.columns if col not in ['TIMESTAMP', 'RH_Aasu']]
# # Align features
# X_pred = df_pred_fe[X_model_cols]

# # Predict with trained model (uses tuned hyperparameters)
# df_pred_fe['RH_Aasu_predicted'] = model.predict(X_pred)

# # Export
# df_pred_fe[['TIMESTAMP', 'RH_Aasu_predicted']].to_csv("aasu_rh_filled.csv", index=False)
