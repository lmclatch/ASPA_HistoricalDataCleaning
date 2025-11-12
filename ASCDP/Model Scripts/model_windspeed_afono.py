#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 07:13:23 2025

@author: lizamclatchy
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import optuna
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


#Using a regression model to adequately predict wind data, not forecasting, learning and predicting missing
combined_df = pd.read_csv("/Users/lizamclatchy/ASCDP/Data Cleaning/Cleaned Model Input Data/train_vaipito_WS_mph_S_WVT.csv")
selected_columns = ['TIMESTAMP', 'WS_mph_S_WVT'] + [col for col in combined_df.columns if col not in ['TIMESTAMP', 'WS_mph_S_WVT']]
rh_data = combined_df[selected_columns].copy()
rh_data = rh_data.dropna(subset=['WS_mph_S_WVT'])  # Keep only rows where RH_Aasu is not NaN

def feature_engineering(df):
    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    target_column = 'WS_mph_S_WVT'
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
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

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

    # Heat Index (approximation using air temp and RH)
    if 'AirTF_target' in df.columns and 'RH_target' in df.columns:
        df['heat_index_target'] = df['AirTF_target'] * df['RH_target'] / 100
        df['temp_trend_1h_target'] = df['AirTF_target'].rolling(6).apply(lambda x: x.iloc[-1] - x.iloc[0], raw=False)
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.interpolate(method='linear', limit_direction='both')

    
    # Now add interaction terms here (currently commented out)
    df['wind_per_rh'] = df['wind_speed_weighted_0_rolling6'] / (df['relative_humidity_weighted_0_rolling6'] + 1e-3)
    df['solar_per_temp'] = df['SolarMJ_target_rolling4'] / (df['air_temp_weighted_0_rolling6'] + 1e-3)
    
    return df
    

 
# Safe SMAPE calculation to handle zero values
def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

def cross_validate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_scores, val_scores = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)

        train_mae = mean_absolute_error(y_tr, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)

        train_scores.append(train_mae)
        val_scores.append(val_mae)

        print(f"Fold {len(train_scores)} - Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}")

    return train_scores, val_scores

def prepare_train_test_data(df, target_column, test_size=0.2):
    df = df.sort_values(by="TIMESTAMP")
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)
    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)
    X_train = train_df.drop(columns=[target_column, 'TIMESTAMP'])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column, 'TIMESTAMP'])
    y_test = test_df[target_column]
    return X_train, X_test, y_train, y_test

target_column = 'WS_mph_S_WVT'

X_train, X_test, y_train, y_test = prepare_train_test_data(rh_data, target_column)

def xgboost_regression(X_train, X_test, y_train, y_test, target_column, n_splits=10):

    # --- 3. Define Objective Function for Optuna with CV ---
    def objective(trial):
        params = {
           'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
           'max_depth': trial.suggest_int('max_depth', 3, 15),
           'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
           'subsample': trial.suggest_float('subsample', 0.6, 1.0),
           'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
           'gamma': trial.suggest_float('gamma', 0, 1),
           'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
           'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
           'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
           'objective': 'reg:squarederror',
           'random_state': 42
       }

        model = XGBRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
    
        for train_index, val_index in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))
    
        return np.mean(mae_scores)

    # --- 4. Run Optuna ---
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print("Best Params:", best_params)

    # --- 5. Cross-Validate Best Model and Default Model ---
    best_model = XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    default_model = XGBRegressor(objective='reg:squarederror', random_state=42)

    train_mae_best, val_mae_best = cross_validate_model(best_model, X_train, y_train, n_splits=n_splits)
    train_mae_default, val_mae_default = cross_validate_model(default_model, X_train, y_train, n_splits=n_splits)

    # --- 6. Plot CV Results ---
    folds = range(1, len(train_mae_best) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(folds, train_mae_default, 'o-', label='Default Train MAE')
    plt.plot(folds, val_mae_default, 'o--', label='Default Val MAE')
    plt.plot(folds, train_mae_best, 's-', label='Tuned Train MAE')
    plt.plot(folds, val_mae_best, 's--', label='Tuned Val MAE')
    plt.xlabel("Fold")
    plt.ylabel("MAE")
    plt.title("Train vs Validation MAE per Fold (Default vs Tuned)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- 7. Fit Final Model and Evaluate ---
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_train, y_train)
    print("MAE (Linear):", mean_absolute_error(y_test, lr.predict(X_test)))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MAE: {mae:.4f} mph")
    print(f"R²:    {r2:.4f}")

    # --- 8. Plot Final Prediction ---
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--', color='red')
    plt.title(f'Actual vs Predicted Wind Speed ({target_column})')
    plt.xlabel("Index")
    plt.ylabel("Wind Speed (log-transformed)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    from xgboost import plot_importance

    plt.figure(figsize=(10, 6))
    plot_importance(best_model, importance_type='gain', max_num_features=15, title='XGBoost Feature Importance')
    plt.tight_layout()
    plt.show()
    
    
    return best_model, y_test, y_pred, best_params



def lightgbm_regression(X_train, X_test, y_train, y_test, target_column, n_splits=10):

    # --- 2. Define Optuna Objective ---
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 0.8),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
            'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
            'random_state': 42
        }
        model = LGBMRegressor(**params)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
        for train_index, val_index in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae_scores.append(mean_absolute_error(y_val, preds))
        return np.mean(mae_scores)

    # --- 3. Run Optuna ---
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print("Best Params (LGBM):", best_params)

    # --- 4. Fit Best Model and Evaluate ---
    best_model = LGBMRegressor(**best_params,objective='quantile', alpha=0.5, random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)


    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"LGBM MAE: {mae:.4f}")
    print(f"LGBM R²: {r2:.4f}")

    # --- 5. Plot Results ---
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', linestyle='--', color='green')
    plt.title(f'Actual vs Predicted ({target_column}) - LightGBM')
    plt.xlabel("Index")
    plt.ylabel(target_column)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return best_model, y_test, y_pred, best_params

model_xgb, _, _, best_params = xgboost_regression(X_train, X_test, y_train, y_test, target_column)
model_lgbm, _, _, best_params_lgbm = lightgbm_regression(X_train, X_test, y_train, y_test,target_column)

estimators = [
    ('xgb', XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)),
    ('lgb', LGBMRegressor(**best_params_lgbm, objective='quantile', alpha=0.5, random_state=42)),
]

from sklearn.ensemble import GradientBoostingRegressor
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
    passthrough=True,
    n_jobs=-1
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

# Predictions
y_pred_xgb  = model_xgb.predict(X_test)
y_pred_lgbm = model_lgbm.predict(X_test)
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
# Metrics table
rows = []
for name, pred in [
    ("XGBoost",   y_pred_xgb),
    ("LightGBM",  y_pred_lgbm),
    ("Stacked",   y_pred_stack),
]:
    mae  = mean_absolute_error(y_test, pred)
    rmse_val = rmse(y_test, pred)           # <-- here
    r2   = r2_score(y_test, pred)
    mape = safe_mape(y_test, pred)
    rows.append({"model": name, "MAE": mae, "RMSE": rmse_val, "R2": r2, "MAPE_%": mape})

metrics_df = pd.DataFrame(rows)

# Save (change path/name as you like)
out_path = "/Users/lizamclatchy/ASCDP/Results Analysis/WS_mph_S_WVT_vaipito_model_error_metrics.csv"
metrics_df.to_csv(out_path, index=False)

print(f"Saved model metrics to: {out_path}")
print(metrics_df)

