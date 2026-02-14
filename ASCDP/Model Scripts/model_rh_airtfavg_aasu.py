#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 08:24:39 2025

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

#Set seed
import random
random.seed(42)        
np.random.seed(42)

#Using a regression model to adequately predict wind data, not forecasting, learning and predicting missing
combined_df = pd.read_csv("/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Model Input Data/train_afono_WindDir_SD1_WVT.csv")
selected_columns = ['TIMESTAMP', 'WindDir_SD1_WVT'] + [col for col in combined_df.columns if col not in ['TIMESTAMP', 'WindDir_SD1_WVT']]
rh_data = combined_df[selected_columns].copy()
rh_data = rh_data.dropna(subset=['WindDir_SD1_WVT'])  # Keep only rows where RH_Aasu is not NaN
#rh_data = rh_data[rh_data.index <= 2500] only for Airtf_avg_aasu



def feature_engineering(df):
    df = df.copy()
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
  
    target_column = 'WindDir_SD1_WVT'
    feature_cols = [col for col in df.columns if col not in ['TIMESTAMP', target_column,'Elevation_target','synoptic_elevation_1','synoptic_elevation_0']]    
    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag6'] = df[col].shift(6)
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
    #df['wind_per_rh'] = df['wind_speed_weighted_0_rolling6'] / (df['relative_humidity_weighted_0_rolling6'] + 1e-3)
   # df['solar_per_temp'] = df['SolarMJ_target_rolling6'] / (df['AirTF_target_rolling6'] + 1e-3)
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
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]
    return X_train, X_test, y_train, y_test

target_column = 'WindDir_SD1_WVT'

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
            mae_scores.append(mean_absolute_error(y_val, preds))
    
        return np.mean(mae_scores)

    # --- 4. Run Optuna ---
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
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
    #y_pred = np.clip(y_pred, a_min=0, a_max=None)

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
    plt.ylabel("Rainfall")
    plt.legend()
    plt.tight_layout()
    plt.show()
    from xgboost import plot_importance

    plt.figure(figsize=(10, 6))
    plot_importance(best_model, importance_type='gain', max_num_features=15, title='XGBoost Feature Importance: Rainfall')
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
        model = LGBMRegressor(objective='mae', **params)
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
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print("Best Params (LGBM):", best_params)

    # --- 4. Fit Best Model and Evaluate ---
    best_model = LGBMRegressor(**best_params,objective='mae', random_state=42)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    #y_pred = np.clip(y_pred, a_min=0, a_max=None)


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
    ('lgb', LGBMRegressor(**best_params_lgbm, objective='mae',random_state=42)),
]

from sklearn.ensemble import GradientBoostingRegressor
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
    passthrough=True,
    #n_jobs=-1
)

stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
#for rain
#y_pred_stack = np.clip(stack.predict(X_test), 0, None)

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < eps, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) *     100.0)

# Predictions
y_pred_xgb  = model_xgb.predict(X_test)
y_pred_lgbm = model_lgbm.predict(X_test)


# #For Rain_in_Tot_Aasu
# y_pred_xgb   = np.clip(model_xgb.predict(X_test), 0, None)
# y_pred_lgbm  = np.clip(model_lgbm.predict(X_test), 0, None)
# y_pred_stack = np.clip(stack.predict(X_test), 0, None)

# Metrics table
rows = []
for name, pred in [
    ("XGBoost",   y_pred_xgb),
    ("LightGBM",  y_pred_lgbm),
    ("Stacked",   y_pred_stack),
]:
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    mape = safe_mape(y_test, pred)
    rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape})

metrics_df = pd.DataFrame(rows)

# Save (change path/name as you like)
out_path = "/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Results Analysis/Afono_WindDir_SD1_WVT_error_metrics.csv"
metrics_df.to_csv(out_path, index=False)

#print(f"Saved model metrics to: {out_path}")
print(metrics_df)
###SAVE MODELS

import joblib
joblib.dump(model_xgb, 'afono_windsd1_model_xgb.pkl')
joblib.dump(model_lgbm, 'afono_wind_sd1model_lgbm.pkl')
joblib.dump(stack, 'afono_windsd1_stack.pkl')


import re
import textwrap
import numpy as np
import matplotlib.pyplot as plt

# --- DISCRETE HEC palette in your custom order ---
HEC_DISCRETE = [
    "#5F4690FF",  # deep purple
    "#1D6996FF",  # blue
    "#38A6A5FF",  # teal
    "#0F8554FF",  # green
    "#73AF48FF",  # light green
    "#EDAD08FF",  # yellow
    "#E17C05FF",  # orange
    "#CC503EFF",  # red-orange
    "#94346EFF",  # magenta
    "#6F4070FF",  # mauve
    "#994E95FF",  # purple-pink
    "#666666FF",  # gray
]

# --- Base rename map (covers bases; lag/rolling handled automatically) ---
rename_map = {
    # Wind (weighted)
    "wind_speed_weighted_0": "Pago Pago weighted wind speed",
    "wind_speed_weighted_1": "Siufaga Ridge weighted wind speed",

    # Air temp / RH (weighted)
    "air_temp_weighted_0": "Pago Pago weighted air temperature",
    "air_temp_weighted_1": "Siufaga Ridge weighted air temperature",
    "relative_humidity_weighted_0": "Pago Pago weighted relative humidity",
    "relative_humidity_weighted_1": "Siufaga Ridge weighted relative humidity",

    # Target station vars
    "PTemp_target": "Max. temperature of target station",
    "AirTF_target": "Air temperature of target station",
    "RH_target": "Relative humidity of target station",
    "SolarW_target": "Solar radiation of target station (W/m²)",
    "SolarMJ_target": "Solar energy of target station (MJ/m²)",

    # Wind direction encodings (weighted)
    "wind_direction_sin_weighted_0": "Pago Pago weighted wind direction (sin)",
    "wind_direction_cos_weighted_0": "Pago Pago weighted wind direction (cos)",
    "wind_direction_sin_weighted_1": "Siufaga Ridge weighted wind direction (sin)",
    "wind_direction_cos_weighted_1": "Siufaga Ridge weighted wind direction (cos)",

    # Calendar / indicators
    "month": "Month",
    "Day_of_week": "Day of week",
    "Season_summer": "Summer",
    "Season_winter": "Winter",
    "is_daytime": "Daytime",

    # Other
    "solar_per_temp": "Solar / Temp",
}

# ---------- label helpers ----------
def sentence_case(s: str) -> str:
    s = (s or "").strip()
    return s[:1].upper() + s[1:] if s else s

def wrap_label(s: str, width: int = 28, max_lines=None) -> str:
    """
    Wrap label text to a given width.
    If max_lines is None, keep all lines (no truncation).
    """
    lines = textwrap.wrap(s, width=width)
    if max_lines is not None:
        lines = lines[:max_lines]
    return "\n".join(lines)

def _mins_to_pretty(mins: int) -> str:
    h, m = divmod(int(mins), 60)
    if h and m:
        return f"{h}h {m}m"
    if h:
        return f"{h}h"
    return f"{m}m"

def annotate_lag_rolling_from_shifts(label: str, step_minutes: int = 15) -> str:
    """
    Your 15-min feature engineering:
      lag1 = shift(2)  => 30m
      lag3 = shift(5)  => 1h 15m
      lag6 = shift(7)  => 1h 45m
    Rolling windows:
      rolling2 => 30m, rolling4 => 1h, rolling6 => 1h 30m
    """
    out = label

    lag_label_to_shift_steps = {"1": 2, "3": 5, "6": 7}

    def repl_lag(m):
        lag_label = m.group(2)
        steps = lag_label_to_shift_steps.get(lag_label, int(lag_label))
        mins = steps * step_minutes
        return f"{m.group(1)}{lag_label} ({_mins_to_pretty(mins)})"

    out = re.sub(r"\b(lag\s+)(\d+)\b(?!\s*\()", repl_lag, out, flags=re.IGNORECASE)

    def repl_roll(m):
        n = int(m.group(2))
        mins = n * step_minutes
        return f"{m.group(1)}{n} ({_mins_to_pretty(mins)})"

    out = re.sub(r"\b(rolling\s+)(\d+)\b(?!\s*\()", repl_roll, out, flags=re.IGNORECASE)
    return out

# --- base fallback renaming for *_lag# and *_rolling# ---
def rename_one_feature(raw_key: str, rename_map: dict) -> str:
    """
    Robust renamer:
    - exact match on raw_key
    - case/whitespace-insensitive lookup
    - base-key fallback for *_lag# and *_rolling#
    """
    if not rename_map:
        return raw_key

    # normalised map: strip + lowercase
    norm_map = {k.strip().lower(): v for k, v in rename_map.items()}

    # 1) exact match
    if raw_key in rename_map:
        return rename_map[raw_key]

    # 2) case/whitespace-insensitive match
    key_clean = raw_key.strip().lower()
    if key_clean in norm_map:
        return norm_map[key_clean]

    # 3) base fallback for *_lag# / *_rolling#
    base_raw = re.sub(r"_(rolling|lag)\d+$", "", raw_key)
    base_clean = base_raw.strip().lower()

    if base_raw in rename_map:
        base_label = rename_map[base_raw]
    elif base_clean in norm_map:
        base_label = norm_map[base_clean]
    else:
        return raw_key  # nothing matched

    # build suffix from raw key ("rolling2" -> "rolling 2")
    suffix = raw_key[len(base_raw):].lstrip("_")
    suffix = re.sub(r"(rolling|lag)(\d+)", r"\1 \2", suffix, flags=re.IGNORECASE)
    if suffix:
        return f"{base_label}, {suffix}"
    return base_label

def rename_features(features,
                    rename_map=None,
                    step_minutes: int = 15,
                    wrap_width: int = 28,
                    max_lines: int = 3):
    """
    Rename raw feature keys, annotate lag/rolling with time (e.g. 30m),
    sentence-case them, and wrap into up to max_lines lines.
    """
    if rename_map is None:
        renamed = list(features)
    else:
        renamed = [rename_one_feature(f, rename_map) for f in features]

    renamed = [annotate_lag_rolling_from_shifts(s, step_minutes=step_minutes) for s in renamed]
    renamed = [sentence_case(s) for s in renamed]
    renamed = [wrap_label(s, width=wrap_width, max_lines=max_lines) for s in renamed]
    return renamed

# ---------- plotting with normalized gain (% of total) ----------
def plot_feature_importance_discrete(
    model,
    model_type,
    feature_names,
    max_features=10,
    importance_type="gain",
    title=None,
    high_is_dark=True,     # "highest importance gets earliest palette color"
    rename_map=None,
    palette=HEC_DISCRETE,
    step_minutes: int = 15,
    wrap_width: int = 28,
    max_lines: int = 3,
    tick_fontsize: int = 9,
    label_fontsize: int = 10,
    title_fontsize: int = 12,
    normalize: str = "sum",   # "sum" -> % of total gain
):
    # ---- extract (feature, importance) pairs ----
    if model_type.lower() == "xgboost":
        score = model.get_booster().get_score(importance_type=importance_type)
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

    # Reverse for barh so largest ends up at top
    feat_raw = [p[0] for p in pairs][::-1]
    imp = np.array([p[1] for p in pairs][::-1], dtype=float)

    # --- normalize gain ---
    xlabel = f"Importance ({importance_type})"
    if normalize == "sum":
        total = imp.sum()
        if total > 0:
            imp = imp / total * 100.0
            xlabel = "Relative importance (% of total gain)"
    elif normalize == "max":
        m = imp.max()
        if m > 0:
            imp = imp / m
            xlabel = "Relative importance (0–1, normalized)"

    # rename + annotate + wrap
    feat = rename_features(
        feat_raw,
        rename_map=rename_map,
        step_minutes=step_minutes,
        wrap_width=wrap_width,
        max_lines=max_lines,
    )

    # discrete rank -> color (no interpolation)
    ranks_desc = np.arange(len(pairs))  # 0=highest
    if not high_is_dark:
        ranks_desc = ranks_desc[::-1]
    ranks_for_plot = ranks_desc[::-1]   # match barh order
    colors = [palette[i % len(palette)] for i in ranks_for_plot]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat, imp, color=colors)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_title(title or f"{model_type.capitalize()} Feature Importance (top {max_features})",
                 fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)

    # add % labels on bars when normalized by sum
    if normalize == "sum":
        max_imp = max(imp) if len(imp) else 0
        for bar, val in zip(bars, imp):
            ax.text(
                bar.get_width() + max_imp * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                ha="left",
                fontsize=tick_fontsize,
            )

    fig.tight_layout()
    # extra room on the left for long multi-line labels
    fig.subplots_adjust(left=0.35)
    plt.show()

# --- Usage example for this variable (Std of Wind Direction at Poloa) ---

TITLE = "Std of Wind Direction (\N{DEGREE SIGN}) Afono"

plot_feature_importance_discrete(
    model_lgbm,
    model_type="lightgbm",
    feature_names=X_train.columns,
    max_features=10,
    importance_type="gain",
    title=f"LightGBM: {TITLE}",
    rename_map=rename_map,
    normalize="sum",
)

plot_feature_importance_discrete(
    model_xgb,
    model_type="xgboost",
    feature_names=X_train.columns,
    max_features=10,
    importance_type="gain",
    title=f"XGBoost: {TITLE}",
    rename_map=rename_map,
    normalize="sum",
)



