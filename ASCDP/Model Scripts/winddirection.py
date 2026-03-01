#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: lizamclatchy
"""

import re
import textwrap
import random
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import optuna

# ── Reproducibility ────────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ==============================================================================
# 0) Circular utilities + angular metrics
# ==============================================================================

def deg_to_sin_cos(x_deg):
    x = pd.to_numeric(x_deg, errors="coerce")
    th = np.deg2rad(x)
    return np.sin(th), np.cos(th)

def sin_cos_to_deg(sin_vals, cos_vals):
    th = np.rad2deg(np.arctan2(sin_vals, cos_vals))
    return (th + 360) % 360

def _angular_error(y_true_deg, y_pred_deg):
    y_true_deg = np.asarray(y_true_deg, dtype=float)
    y_pred_deg = np.asarray(y_pred_deg, dtype=float)
    return np.abs((y_pred_deg - y_true_deg + 180.0) % 360.0 - 180.0)

def angular_mae(y_true_deg, y_pred_deg):
    return float(np.mean(_angular_error(y_true_deg, y_pred_deg)))

def angular_rmse(y_true_deg, y_pred_deg):
    e = _angular_error(y_true_deg, y_pred_deg)
    return float(np.sqrt(np.mean(e**2)))

# ==============================================================================
# 1) Data loading
# ==============================================================================

#CHANGE THIS BASED ON STATION
combined_df = pd.read_csv(
    "/Users/lizamclatchy/Documents/GitHub/ASPA_HistoricalDataCleaning/ASCDP/"
    "Data Cleaning/Cleaned Model Input Data/train_poloa_WindDir_D1_WVT.csv"
)

TARGET_DEG  = "WindDir_D1_WVT"   # raw degrees column (kept for train/pred splits)
TARGET_SIN  = "WindDir_sin"
TARGET_COS  = "WindDir_cos"

# Keep only rows where the target exists
rh_data = combined_df.dropna(subset=[TARGET_DEG]).copy()

# ==============================================================================
# 2) Feature engineering
# ==============================================================================

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])

    # Columns to lag/roll: everything except identifiers, targets, and elevations
    skip = {"TIMESTAMP", TARGET_DEG, TARGET_SIN, TARGET_COS,
            "Elevation_target", "synoptic_elevation_0", "synoptic_elevation_1"}
    feature_cols = [c for c in df.columns if c not in skip]

    for col in feature_cols:
        df[f"{col}_lag1"]     = df[col].shift(1)
        df[f"{col}_lag3"]     = df[col].shift(3)
        df[f"{col}_lag6"]     = df[col].shift(6)
        df[f"{col}_rolling2"] = df[col].rolling(2).mean()
        df[f"{col}_rolling4"] = df[col].rolling(4).mean()
        df[f"{col}_rolling6"] = df[col].rolling(6).mean()

    # Time features
    df["hour_of_day"] = df["TIMESTAMP"].dt.hour
    df["is_daytime"]  = ((df["hour_of_day"] >= 6) & (df["hour_of_day"] <= 18)).astype(int)
    df["day_of_week"] = df["TIMESTAMP"].dt.dayofweek
    df["month"]       = df["TIMESTAMP"].dt.month

    # Season one-hot (fixed categories so train/test always match)
    def _season(m):
        if m in [12, 1, 2]:  return "winter"
        if m in [3, 4, 5]:   return "spring"
        if m in [6, 7, 8]:   return "summer"
        return "fall"

    df["season"] = pd.Categorical(
        df["month"].apply(_season),
        categories=["winter", "spring", "summer", "fall"]
    )
    season_dummies = pd.get_dummies(df["season"], prefix="season").astype(int)
    df = pd.concat([df, season_dummies], axis=1).drop(columns=["season"])

    # Ensure all non-TIMESTAMP columns are numeric
    for col in df.columns:
        if col != "TIMESTAMP":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

# ==============================================================================
# 3) Train / test split  (temporal — no shuffle)
# ==============================================================================

def prepare_train_test_data(df: pd.DataFrame, test_size: float = 0.2):
    df = df.sort_values("TIMESTAMP")
    train_df, test_df = train_test_split(df, test_size=test_size, shuffle=False)

    train_df = feature_engineering(train_df)
    test_df  = feature_engineering(test_df)

    drop_cols = {"TIMESTAMP", TARGET_DEG, TARGET_SIN, TARGET_COS}

    def _split(d):
        X = d.drop(columns=[c for c in drop_cols if c in d.columns]).dropna()
        y_sin = d[TARGET_SIN].loc[X.index]
        y_cos = d[TARGET_COS].loc[X.index]
        y_deg = d[TARGET_DEG].loc[X.index]
        return X, y_sin, y_cos, y_deg

    X_train, y_train_sin, y_train_cos, y_train_deg = _split(train_df)
    X_test,  y_test_sin,  y_test_cos,  y_test_deg  = _split(test_df)

    return X_train, X_test, y_train_sin, y_train_cos, y_train_deg, y_test_sin, y_test_cos, y_test_deg


X_train, X_test, y_train_sin, y_train_cos, y_train_deg, y_test_sin, y_test_cos, y_test_deg = \
    prepare_train_test_data(rh_data)

# ==============================================================================
# 4) Cross-validation helper (angular error)
# ==============================================================================

def cross_validate_dual(model_sin, model_cos, X, y_sin, y_cos, y_deg, n_splits=5):
    """TimeSeriesSplit CV reporting angular MAE per fold."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    val_angular_maes = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr,  X_val  = X.iloc[tr_idx],     X.iloc[val_idx]
        ys_tr, ys_val = y_sin.iloc[tr_idx],  y_sin.iloc[val_idx]
        yc_tr, yc_val = y_cos.iloc[tr_idx],  y_cos.iloc[val_idx]
        yd_val        = y_deg.iloc[val_idx]

        model_sin.fit(X_tr, ys_tr)
        model_cos.fit(X_tr, yc_tr)

        pred_sin = model_sin.predict(X_val)
        pred_cos = model_cos.predict(X_val)
        pred_deg = sin_cos_to_deg(pred_sin, pred_cos)

        ang_mae = angular_mae(yd_val.values, pred_deg)
        val_angular_maes.append(ang_mae)
        print(f"  Fold {fold} — Angular MAE: {ang_mae:.2f}°")

    print(f"  Mean Angular MAE: {np.mean(val_angular_maes):.2f}°")
    return val_angular_maes

# ==============================================================================
# 5) XGBoost — dual model (sin + cos)
# ==============================================================================

def xgboost_regression_winddir(X_train, X_test,
                                y_train_sin, y_train_cos,
                                y_test_sin,  y_test_cos, y_test_deg,
                                n_splits=10):

    def _objective(trial, y_target):
        params = {
            "n_estimators":    trial.suggest_int("n_estimators",    100, 1000),
            "max_depth":       trial.suggest_int("max_depth",         3,   15),
            "learning_rate":   trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample":       trial.suggest_float("subsample",       0.6,  1.0),
            "colsample_bytree":trial.suggest_float("colsample_bytree",0.5,  1.0),
            "gamma":           trial.suggest_float("gamma",           0,    1),
            "min_child_weight":trial.suggest_int("min_child_weight",  1,   10),
            "lambda":          trial.suggest_float("lambda",    1e-3, 10.0, log=True),
            "alpha":           trial.suggest_float("alpha",     1e-3, 10.0, log=True),
            "objective": "reg:squarederror",
            "random_state": 42,
        }
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, val in tscv.split(X_train):
            m = XGBRegressor(**params)
            m.fit(X_train.iloc[tr], y_target.iloc[tr])
            scores.append(mean_absolute_error(y_target.iloc[val], m.predict(X_train.iloc[val])))
        return np.mean(scores)

    print("── XGBoost: tuning sin model ──")
    study_sin = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_sin.optimize(lambda t: _objective(t, y_train_sin), n_trials=100, show_progress_bar=True)

    print("── XGBoost: tuning cos model ──")
    study_cos = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cos.optimize(lambda t: _objective(t, y_train_cos), n_trials=100, show_progress_bar=True)

    best_params_sin = study_sin.best_params
    best_params_cos = study_cos.best_params
    print("Best params (sin):", best_params_sin)
    print("Best params (cos):", best_params_cos)

    # Fit final models
    model_sin = XGBRegressor(**best_params_sin, objective="reg:squarederror", random_state=42)
    model_cos = XGBRegressor(**best_params_cos, objective="reg:squarederror", random_state=42)
    model_sin.fit(X_train, y_train_sin)
    model_cos.fit(X_train, y_train_cos)

    # Reconstruct degrees
    pred_sin = model_sin.predict(X_test)
    pred_cos = model_cos.predict(X_test)
    pred_deg = sin_cos_to_deg(pred_sin, pred_cos)

    ang_mae  = angular_mae(y_test_deg.values, pred_deg)
    ang_rmse = angular_rmse(y_test_deg.values, pred_deg)
    print(f"XGBoost — Angular MAE: {ang_mae:.2f}°   Angular RMSE: {ang_rmse:.2f}°")

    # Plot
    plt.figure(figsize=(14, 4))
    plt.plot(y_test_deg.values, label="Actual", color="blue", alpha=0.7)
    plt.plot(pred_deg,          label="Predicted", linestyle="--", color="red", alpha=0.7)
    plt.title(f"XGBoost — Actual vs Predicted Wind Direction ({TARGET_DEG})")
    plt.xlabel("Index")
    plt.ylabel("Wind Direction (°)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model_sin, model_cos, pred_deg, best_params_sin, best_params_cos

# ==============================================================================
# 6) LightGBM — dual model (sin + cos)
# ==============================================================================

def lightgbm_regression_winddir(X_train, X_test,
                                 y_train_sin, y_train_cos,
                                 y_test_sin,  y_test_cos, y_test_deg,
                                 n_splits=10):

    def _objective(trial, y_target):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators",      100, 500),
            "max_depth":         trial.suggest_int("max_depth",            3,  10),
            "learning_rate":     trial.suggest_float("learning_rate",   0.01, 0.1, log=True),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 0.8),
            "num_leaves":        trial.suggest_int("num_leaves",          20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples",   10,  50),
            "lambda_l1":         trial.suggest_float("lambda_l1",          0,  10),
            "lambda_l2":         trial.suggest_float("lambda_l2",          0,  10),
            "random_state": 42,
        }
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, val in tscv.split(X_train):
            m = LGBMRegressor(objective="mae", **params)
            m.fit(X_train.iloc[tr], y_target.iloc[tr])
            scores.append(mean_absolute_error(y_target.iloc[val], m.predict(X_train.iloc[val])))
        return np.mean(scores)

    print("── LightGBM: tuning sin model ──")
    study_sin = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_sin.optimize(lambda t: _objective(t, y_train_sin), n_trials=100, show_progress_bar=True)

    print("── LightGBM: tuning cos model ──")
    study_cos = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
    study_cos.optimize(lambda t: _objective(t, y_train_cos), n_trials=100, show_progress_bar=True)

    best_params_sin = study_sin.best_params
    best_params_cos = study_cos.best_params
    print("Best params LGBM (sin):", best_params_sin)
    print("Best params LGBM (cos):", best_params_cos)

    model_sin = LGBMRegressor(**best_params_sin, objective="mae", random_state=42)
    model_cos = LGBMRegressor(**best_params_cos, objective="mae", random_state=42)
    model_sin.fit(X_train, y_train_sin)
    model_cos.fit(X_train, y_train_cos)

    pred_sin = model_sin.predict(X_test)
    pred_cos = model_cos.predict(X_test)
    pred_deg = sin_cos_to_deg(pred_sin, pred_cos)

    ang_mae  = angular_mae(y_test_deg.values, pred_deg)
    ang_rmse = angular_rmse(y_test_deg.values, pred_deg)
    print(f"LightGBM — Angular MAE: {ang_mae:.2f}°   Angular RMSE: {ang_rmse:.2f}°")

    plt.figure(figsize=(14, 4))
    plt.plot(y_test_deg.values, label="Actual", color="blue", alpha=0.7)
    plt.plot(pred_deg,          label="Predicted", linestyle="--", color="green", alpha=0.7)
    plt.title(f"LightGBM — Actual vs Predicted Wind Direction ({TARGET_DEG})")
    plt.xlabel("Index")
    plt.ylabel("Wind Direction (°)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return model_sin, model_cos, pred_deg, best_params_sin, best_params_cos

# ==============================================================================
# 7) Run models
# ==============================================================================

xgb_sin, xgb_cos, xgb_pred_deg, xgb_params_sin, xgb_params_cos = xgboost_regression_winddir(
    X_train, X_test,
    y_train_sin, y_train_cos,
    y_test_sin, y_test_cos, y_test_deg
)

lgbm_sin, lgbm_cos, lgbm_pred_deg, lgbm_params_sin, lgbm_params_cos = lightgbm_regression_winddir(
    X_train, X_test,
    y_train_sin, y_train_cos,
    y_test_sin, y_test_cos, y_test_deg
)

# ==============================================================================
# 8) Stacked ensemble  (stacks on sin and cos separately, then reconstruct)
# ==============================================================================

def make_stack(params_sin, params_cos):
    stack_sin = StackingRegressor(
        estimators=[
            ("xgb",  XGBRegressor(**params_sin,  objective="reg:squarederror", random_state=42)),
            ("lgbm", LGBMRegressor(**lgbm_params_sin, objective="mae",         random_state=42)),
        ],
        final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
        passthrough=True,
    )
    stack_cos = StackingRegressor(
        estimators=[
            ("xgb",  XGBRegressor(**params_cos,  objective="reg:squarederror", random_state=42)),
            ("lgbm", LGBMRegressor(**lgbm_params_cos, objective="mae",         random_state=42)),
        ],
        final_estimator=GradientBoostingRegressor(n_estimators=100, random_state=42),
        passthrough=True,
    )
    return stack_sin, stack_cos

stack_sin, stack_cos = make_stack(xgb_params_sin, xgb_params_cos)
stack_sin.fit(X_train, y_train_sin)
stack_cos.fit(X_train, y_train_cos)

stack_pred_deg = sin_cos_to_deg(stack_sin.predict(X_test), stack_cos.predict(X_test))

# ==============================================================================
# 9) Metrics table
# ==============================================================================

rows = []
for name, pred in [("XGBoost", xgb_pred_deg), ("LightGBM", lgbm_pred_deg), ("Stacked", stack_pred_deg)]:
    rows.append({
        "model":        name,
        "Angular_MAE":  angular_mae(y_test_deg.values, pred),
        "Angular_RMSE": angular_rmse(y_test_deg.values, pred),
        "R2":           r2_score(y_test_deg.values, pred),
    })
metrics_df = pd.DataFrame(rows)
print(metrics_df)

# CHANGE PATH/NAME
out_path = (
    "/Users/lizamclatchy/Documents/GitHub/ASPA_HistoricalDataCleaning/ASCDP/"
    "Results Analysis/WindDir_D1_WVT_poloa_error_metrics.csv"
)
metrics_df.to_csv(out_path, index=False)

# ==============================================================================
# 10) Save models
# ==============================================================================

joblib.dump(xgb_sin,  "WindDir_D1_WVT_poloa_xgb_sin.pkl")
joblib.dump(xgb_cos,  "WindDir_D1_WVT_poloa_xgb_cos.pkl")
joblib.dump(lgbm_sin, "WindDir_D1_WVT_poloa_lgbm_sin.pkl")
joblib.dump(lgbm_cos, "WindDir_D1_WVT_poloa_lgbm_cos.pkl")
joblib.dump(stack_sin,"WindDir_D1_WVT_poloa_stack_sin.pkl")
joblib.dump(stack_cos,"WindDir_D1_WVT_poloa_stack_cos.pkl")

# ==============================================================================
# 11) Feature importance (reused from your original code — unchanged)
# ==============================================================================

HEC_DISCRETE = [
    "#5F4690FF","#1D6996FF","#38A6A5FF","#0F8554FF","#73AF48FF","#EDAD08FF",
    "#E17C05FF","#CC503EFF","#94346EFF","#6F4070FF","#994E95FF","#666666FF",
]

rename_map = {
    "wind_speed_weighted_0":           "Pago Pago weighted wind speed",
    "wind_speed_weighted_1":           "Siufaga Ridge weighted wind speed",
    "air_temp_weighted_0":             "Pago Pago weighted air temperature",
    "air_temp_weighted_1":             "Siufaga Ridge weighted air temperature",
    "relative_humidity_weighted_0":    "Pago Pago weighted relative humidity",
    "relative_humidity_weighted_1":    "Siufaga Ridge weighted relative humidity",
    "PTemp_target":                    "Max. temperature of target station",
    "AirTF_target":                    "Air temperature of target station",
    "RH_target":                       "Relative humidity of target station",
    "SolarW_target":                   "Solar radiation of target station (W/m²)",
    "SolarMJ_target":                  "Solar energy of target station (MJ/m²)",
    "wind_direction_sin_weighted_0":   "Pago Pago weighted wind direction (sin)",
    "wind_direction_cos_weighted_0":   "Pago Pago weighted wind direction (cos)",
    "wind_direction_sin_weighted_1":   "Siufaga Ridge weighted wind direction (sin)",
    "wind_direction_cos_weighted_1":   "Siufaga Ridge weighted wind direction (cos)",
    "month": "Month", "day_of_week": "Day of week",
    "season_summer": "Summer", "season_winter": "Winter",
    "is_daytime": "Daytime",
    # Target sin/cos lags are also features
    "WindDir_sin": "Wind direction (sin)",
    "WindDir_cos": "Wind direction (cos)",
}

def sentence_case(s):
    s = (s or "").strip()
    return s[:1].upper() + s[1:] if s else s

def wrap_label(s, width=28):
    return "\n".join(textwrap.wrap(s, width=width))

def _mins_to_pretty(mins):
    h, m = divmod(int(mins), 60)
    if h and m: return f"{h}h {m}m"
    if h:       return f"{h}h"
    return f"{m}m"

def annotate_lag_rolling(label, step_minutes=15):
    lag_map = {"1": 2, "3": 5, "6": 7}
    def repl_lag(m):
        steps = lag_map.get(m.group(2), int(m.group(2)))
        return f"{m.group(1)}{m.group(2)} ({_mins_to_pretty(steps * step_minutes)})"
    label = re.sub(r"\b(lag\s+)(\d+)\b(?!\s*\()", repl_lag, label, flags=re.IGNORECASE)
    def repl_roll(m):
        return f"{m.group(1)}{m.group(2)} ({_mins_to_pretty(int(m.group(2)) * step_minutes)})"
    label = re.sub(r"\b(rolling\s+)(\d+)\b(?!\s*\()", repl_roll, label, flags=re.IGNORECASE)
    return label

def rename_one_feature(key, rmap):
    norm = {k.strip().lower(): v for k, v in rmap.items()}
    if key in rmap:                      return rmap[key]
    if key.strip().lower() in norm:      return norm[key.strip().lower()]
    base = re.sub(r"_(rolling|lag)\d+$", "", key)
    bl   = base.strip().lower()
    base_label = rmap.get(base) or norm.get(bl)
    if not base_label:                   return key
    suffix = re.sub(r"(rolling|lag)(\d+)", r"\1 \2", key[len(base):].lstrip("_"), flags=re.IGNORECASE)
    return f"{base_label}, {suffix}" if suffix else base_label

def rename_features(features, rmap=None, step_minutes=15, wrap_width=28):
    renamed = [rename_one_feature(f, rmap) if rmap else f for f in features]
    renamed = [annotate_lag_rolling(s, step_minutes) for s in renamed]
    renamed = [sentence_case(s) for s in renamed]
    renamed = [wrap_label(s, wrap_width) for s in renamed]
    return renamed

def plot_feature_importance_discrete(model, model_type, feature_names,
                                     max_features=10, importance_type="gain",
                                     title=None, rename_map=None,
                                     palette=HEC_DISCRETE, normalize="sum"):
    if model_type.lower() == "xgboost":
        score = model.get_booster().get_score(importance_type=importance_type)
        pairs = []
        for k, v in score.items():
            idx   = int(k[1:]) if k.startswith("f") else None
            fname = feature_names[idx] if idx is not None and idx < len(feature_names) else k
            pairs.append((fname, float(v)))
    elif model_type.lower() == "lightgbm":
        vals  = model.booster_.feature_importance(importance_type=importance_type)
        names = model.booster_.feature_name()
        pairs = [(fn, float(v)) for fn, v in zip(names, vals)]
    else:
        raise ValueError("model_type must be 'xgboost' or 'lightgbm'")

    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:max_features]
    feat_raw = [p[0] for p in pairs][::-1]
    imp      = np.array([p[1] for p in pairs][::-1], dtype=float)

    xlabel = f"Importance ({importance_type})"
    if normalize == "sum" and imp.sum() > 0:
        imp    = imp / imp.sum() * 100.0
        xlabel = "Relative importance (% of total gain)"

    feat   = rename_features(feat_raw, rmap=rename_map)
    colors = [palette[i % len(palette)] for i in range(len(pairs))[::-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat, imp, color=colors)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title or f"{model_type.capitalize()} Feature Importance (top {max_features})", fontsize=12)
    ax.tick_params(labelsize=9)

    if normalize == "sum":
        mx = max(imp) if len(imp) else 0
        for bar, val in zip(bars, imp):
            ax.text(bar.get_width() + mx * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", ha="left", fontsize=9)

    fig.tight_layout()
    fig.subplots_adjust(left=0.35)
    plt.show()


TITLE = "Wind Direction (°) Poloa"  # CHANGE as needed

# Plot for sin model (most informative for direction)
plot_feature_importance_discrete(
    lgbm_sin, model_type="lightgbm", feature_names=X_train.columns,
    max_features=10, importance_type="gain",
    title=f"LightGBM (sin): {TITLE}", rename_map=rename_map, normalize="sum",
)
plot_feature_importance_discrete(
    xgb_sin, model_type="xgboost", feature_names=X_train.columns,
    max_features=10, importance_type="gain",
    title=f"XGBoost (sin): {TITLE}", rename_map=rename_map, normalize="sum",
)