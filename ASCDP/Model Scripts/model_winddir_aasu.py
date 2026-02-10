#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 15:52:50 2025

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
combined_df = pd.read_csv("/Users/benslattery/ASCDP/Data Cleaning/Cleaned Model Input Data/train_aasu_WindDir_D1_WVT.csv")
selected_columns = ['TIMESTAMP', 'WindDir_D1_WVT'] + [col for col in combined_df.columns if col not in ['TIMESTAMP', 'WindDir_D1_WVT']]
rh_data = combined_df[selected_columns].copy()


def add_dir_vectors(df, dir_cols, drop_original=True):
    """
    For each direction column in degrees, add _sin and _cos columns.
    """
    df = df.copy()
    for col in dir_cols:
        if col in df.columns:
            x = pd.to_numeric(df[col], errors="coerce")
            theta = np.deg2rad(x)
            df[f"{col}_sin"] = np.sin(theta)
            df[f"{col}_cos"] = np.cos(theta)
            if drop_original:
                df.drop(columns=[col], inplace=True)
    return df
direction_degree_cols = [
  
      'WindDir_D1_WVT'
  
  ]
rh_data = add_dir_vectors(rh_data, direction_degree_cols, drop_original=True)

rh_data = rh_data.dropna(subset=['WindDir_D1_WVT'])  # Keep only rows where RH_Aasu is not NaN

def feature_engineering(df):
    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    target_column = 'WindDir_D1_WVT'

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
    df['wind_per_rh'] = df['wind_speed_weighted_0_rolling6'] / (df['relative_humidity_weighted_0_rolling6'] + 1e-3)
    df['solar_per_temp'] = df['SolarMJ_target_rolling6'] / (df['AirTF_target_rolling6'] + 1e-3)
    return df
# --- Angular error utilities for circular targets (degrees) ---
def _angular_error(y_true_deg, y_pred_deg):
    y_true_deg = np.asarray(y_true_deg, dtype=float)
    y_pred_deg = np.asarray(y_pred_deg, dtype=float)
    return np.abs((y_pred_deg - y_true_deg + 180.0) % 360.0 - 180.0)

def angular_mae(y_true_deg, y_pred_deg):
    return float(np.mean(_angular_error(y_true_deg, y_pred_deg)))

def angular_rmse(y_true_deg, y_pred_deg):
    return float(np.sqrt(np.mean(_angular_error(y_true_deg, y_pred_deg)**2)))

def cross_validate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_scores, val_scores = [], []

    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)

        train_mae = angular_mae(y_tr, train_pred)   # was mean_absolute_error(...)
        val_mae   = angular_mae(y_val, val_pred) 

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
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    return X_train, X_test, y_train, y_test

target_column = 'WindDir_D1_WVT'

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
            model = XGBRegressor(**params)
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae_scores.append(angular_mae(y_val, preds))
    
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

    # from sklearn.linear_model import LinearRegression
    # lr = LinearRegression().fit(X_train, y_train)
    # print("MAE (Linear):", mean_absolute_error(y_test, lr.predict(X_test)))
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # print(f"MAE: {mae:.4f} mph")
    # print(f"R²:    {r2:.4f}")

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
        model = LGBMRegressor(objective='mae',**params)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mae_scores = []
        for train_index, val_index in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            mae_scores.append(angular_mae(y_val, preds))
        return np.mean(mae_scores)

    # --- 3. Run Optuna ---
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print("Best Params (LGBM):", best_params)

    # --- 4. Fit Best Model and Evaluate ---
    best_model = LGBMRegressor(**best_params,objective='mae', random_state=42)
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
    ('lgb', LGBMRegressor(**best_params_lgbm, objective='mae', random_state=42)),
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
    mae  = angular_mae(y_test, pred)
    armse = angular_rmse(y_test, pred)
    r2   = r2_score(y_test, pred)  # keep if you want, but interpret cautiously for circular targets
    rows.append({"model": name, "Angular_MAE_deg": mae, "Angular_RMSE_deg": armse, "R2_linear": r2})

metrics_df = pd.DataFrame(rows)



# Save (change path/name as you like)
out_path = "/Users/benslattery/ASCDP/Results Analysis/WindDir_D1_WVT_aasu_model_error_metrics.csv"
#metrics_df.to_csv(out_path, index=False)

#print(f"Saved model metrics to: {out_path}")
print(metrics_df)



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- 1) Build your HEC colormap (continuous) ---
hec_colors = ['#0E7C7BFF', '#17BEBBFF', '#D4F4DDFF', '#D62246FF', '#4B1D3FFF']
cmap = mcolors.LinearSegmentedColormap.from_list("hec_colors", hec_colors, N=256)

# (Optional) visualize the gradient
plt.figure(figsize=(10, 2))
gradient = np.linspace(0, 1, 300).reshape(1, -1)
plt.imshow(gradient, aspect="auto", cmap=cmap)
plt.title("HEC Color Gradient")
plt.axis("off")
plt.show()

def rename_features(features, rename_map=None):
    """
    Returns a renamed list of feature labels using rename_map.
    Any feature not found in rename_map is returned unchanged.
    """
    if rename_map is None:
        return list(features)
    return [rename_map.get(f, f) for f in features]

# --- 2) Helper to pull importances from a FITTED model and plot with gradient-by-importance ---
def plot_feature_importance_gradient(
    model,
    model_type,
    feature_names,
    max_features=10,
    importance_type="gain",
    title=None,
    high_is_dark=True,
    rename_map=None,   # <-- now supported
):
    # ---- extract (feature, importance) pairs ----
    if model_type.lower() == "xgboost":
        score = model.get_booster().get_score(importance_type=importance_type)  # dict
        pairs = []
        for k, v in score.items():
            if k.startswith("f"):
                idx = int(k[1:])
                fname = feature_names[idx] if idx < len(feature_names) else k
            else:
                fname = k
            pairs.append((fname, float(v)))

    elif model_type.lower() == "lightgbm":
        vals = model.booster_.feature_importance(importance_type=importance_type)
        names = model.booster_.feature_name()
        imp_map = dict(zip(names, vals))
        pairs = [(fn, float(imp_map.get(fn, 0.0))) for fn in feature_names]

    else:
        raise ValueError("model_type must be 'xgboost' or 'lightgbm'")

    # ---- sort + select top features ----
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:max_features]

    # Reverse for barh so largest ends up at top visually
    feat = [p[0] for p in pairs][::-1]
    imp  = np.array([p[1] for p in pairs][::-1], dtype=float)

    # ---- apply renaming via helper ----
    feat = rename_features(feat, rename_map)

    # ---- map importance -> gradient colors ----
    if imp.max() == imp.min():
        normed = np.zeros_like(imp)
    else:
        normed = (imp - imp.min()) / (imp.max() - imp.min())

    t = normed if high_is_dark else (1 - normed)
    bar_colors = [cmap(x) for x in t]

    # ---- plot ----
    plt.figure(figsize=(10, 6))
    plt.barh(feat, imp, color=bar_colors)
    plt.xlabel(f"importance ({importance_type})")
    plt.title(title or f"{model_type.capitalize()} Feature Importance (top {max_features})")
    plt.tight_layout()
    plt.show()


# --- 3) Example usage (NO retraining needed) ---
# Make sure model_xgb and model_lgbm are already fitted, and X_train exists.
rename_map = {
    # Wind (weighted)
    "wind_speed_weighted_0": "Pago Pago Weighted Wind Speed",
    "wind_speed_weighted_0_rolling2": "Pago Pago Weighted Wind Speed, Rolling 2",
    "wind_speed_weighted_0_rolling4": "Pago Pago Weighted Wind Speed, Rolling 4",
    "wind_speed_weighted_0_rolling6": "Pago Pago Weighted Wind Speed, Rolling 6",
    "wind_speed_weighted_0_lag6": "Pago Pago Weighted Wind Speed, Lag 6",

    # Temp (target station)
    "PTemp_target": "Temperature of Target Station",
    "PTemp_target_rolling2": "Temperature of Target Station, Rolling 2",
    "PTemp_target_rolling4": "Temperature of Target Station, Rolling 4",
    "PTemp_target_lag1": "Temperature of Target Station, Lag 1",

    # Other features from your plot
    "month": "Month",
    "solar_per_temp": "Solar / Temp",
    "RH_target": "Relative Humidity of Target Station",

    # Time indicator
    "is_daytime": "Daytime",
}


    
#    "wind_speed_weighted_0_lag1": "Weight'

plot_feature_importance_gradient(
    model_xgb,
    model_type="xgboost",
    feature_names=X_train.columns,
    max_features=10,
    importance_type="gain",
    title="XGBoost Feature Importance: Aasu, WindDir_SD1_WVT",
    high_is_dark=True,
    rename_map=rename_map
)

plot_feature_importance_gradient(
    model_lgbm,
    model_type="lightgbm",
    feature_names=X_train.columns,
    max_features=10,
    importance_type="gain",
    title="LightGBM Feature Importance: Aasu, WindDir_SD1_WVT",
    high_is_dark=True,
    rename_map=rename_map
)