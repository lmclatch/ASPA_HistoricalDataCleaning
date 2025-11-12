#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:37:32 2025

@author: lizamclatchy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 07:37:32 2025

@author: lizamclatchy
"""

from xgboost import XGBRegressor, plot_importance
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import optuna


def create_features(df, target_column, mode="train"):
    df = df.copy()
    feature_cols = [col for col in df.columns if col not in ['TIMESTAMP', 'LAT', 'LON']]

    for col in feature_cols:
        df[f'{col}_lag1'] = df[col].shift(1)
        df[f'{col}_lag3'] = df[col].shift(3)
        df[f'{col}_lag5'] = df[col].shift(5)
        df[f'{col}_rolling2'] = df[col].rolling(window=2).mean()
        df[f'{col}_rolling4'] = df[col].rolling(window=4).mean()
        df[f'{col}_rolling6'] = df[col].rolling(window=6).mean()
            
    if 'AirTF_Avg' in df.columns and 'RH' in df.columns:
        df['HeatIndex_approx'] = df['AirTF_Avg'] * df['RH'] / 100
        df['Air_RH_interaction'] = df['AirTF_Avg'] * df['RH']

# Interaction terms
    df['wind_speed_dir_interaction'] = df['wind_speed_weighted_0'] * df['wind_direction_weighted_0']
    df['wind_speed_dir_interaction_1'] = df['wind_speed_weighted_1'] * df['wind_direction_weighted_1']
    df['target_dir_interaction'] = df[target_column] * df['WindDir_D1_WVT']
    df['target_sd_interaction'] = df[target_column] * df['WindDir_SD1_WVT']
    df['WindDir_D1_WVT_shift'] = df['WindDir_D1_WVT'].diff().abs()
    df['WindDir_SD1_WVT_shift'] = df['WindDir_SD1_WVT'].diff().abs()
    df['DewPoint_approx'] = df['AirTF_Avg'] - ((100 - df['RH']) / 5)
    df['WindSpeed_SD_Interaction'] = df['WS_mph_S_WVT'] * df['WindDir_SD1_WVT']
    df['Wind_Stress_Index'] = df['WS_mph_S_WVT'] ** 2 * df['WindDir_SD1_WVT']
    df['Rain_Temp_Interaction'] = df['Rain_in_Tot'] * df['AirTF_Avg']
    df['Rain_Humidity_Interaction'] = df['Rain_in_Tot'] * df['RH']
    df['hour_of_day'] = pd.to_datetime(df['TIMESTAMP']).dt.hour
    df['is_daytime'] = ((df['hour_of_day'] >= 6) & (df['hour_of_day'] <= 18)).astype(int)
    df['wind_dir_rad'] = np.radians(df['WindDir_D1_WVT'])



    return df


def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100


def forecast_missing(df_train, df_pred, target_column="WS_mph_S_WVT"):
    df_train = df_train[df_train[target_column].notna()].copy()
    df_train = create_features(df_train, target_column=target_column, mode="train")
    df_pred = create_features(df_pred, target_column=target_column, mode="predict")

    # Hold out last 20% of training data for error validation
    train_data, test_data = train_test_split(df_train, test_size=0.2, shuffle=False)

    X_train = train_data.drop(columns=[target_column, 'TIMESTAMP'], errors='ignore')
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column, 'TIMESTAMP'], errors='ignore')
    y_test = test_data[target_column]

    print("\n➡️ Starting Optuna Bayesian Optimization...")
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
        scores = -1 * cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=3, n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)


    # Train with best params
    best_model = XGBRegressor(**study.best_params, objective='reg:squarederror', random_state=42)
    best_model.fit(X_train, y_train)

    # Predict test and pred sets
    y_pred_test = best_model.predict(X_test)
    X_pred = df_pred.drop(columns=[target_column, 'TIMESTAMP'], errors='ignore')

    missing_cols = [col for col in X_train.columns if col not in X_pred.columns]
    for col in missing_cols:
        X_pred[col] = 0
    X_pred = X_pred[X_train.columns]

    if X_pred.empty:
        print("⚠️  X_pred is empty after feature processing. Check df_pred inputs.")
    else:
        df_pred[f'{target_column}_predicted'] = best_model.predict(X_pred)

    test_error = smape(y_test, y_pred_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred_test):.3f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred_test) ** 0.5:.3f}")
    print(f"SMAPE: {test_error:.2f}%")
    print(f"df_pred returned with {len(df_pred)} rows")

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(test_data['TIMESTAMP'], y_test, label='Actual')
    plt.plot(test_data['TIMESTAMP'], y_pred_test, label='Predicted')
    plt.title(f'{target_column}: Predicted vs Actual')
    plt.xlabel('Timestamp')
    plt.ylabel(target_column)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plot_importance(best_model, max_num_features=15)
    plt.title(f'{target_column}: Feature Importances')
    plt.tight_layout()
    plt.show()

    return best_model, df_pred, test_error


# Example use for wind speed
columns_to_predict = ["WS_mph_S_WVT", "WindDir_D1_WVT", "WindDir_SD1_WVT"]

for col in columns_to_predict:
    df_train = pd.read_csv("train_vaipito_windspeed.csv")
    df_pred = pd.read_csv("pred_vaipito_windspeed.csv")
    model, df_pred, err = forecast_missing(df_train, df_pred, target_column=col)
    df_pred.to_csv(f"predicted_{col}.csv", index=False)
    print(f"Completed prediction for {col}\n")

def rolling_forecast_multi(df_train, df_pred, target_columns, steps=48):
    """
    Recursively forecasts missing data for multiple target columns,
    with tracking and optional early exit recovery.
    """
    history = df_train.copy()
    forecasts = []
    smape_logs = {col: [] for col in target_columns}
    iteration_tracker = []  # <-- Track iteration details here

    iteration = 0

    while not df_pred.empty:
        print(f"\n➡️ Recursive iteration {iteration+1}: Forecasting next {steps} steps")
        chunk = df_pred.iloc[:steps].copy()
        chunk_predictions = {}
        iteration_record = {"iteration": iteration + 1}

        # Track start/end timestamps of this chunk
        try:
            iteration_record["start"] = chunk["TIMESTAMP"].iloc[0]
            iteration_record["end"] = chunk["TIMESTAMP"].iloc[-1]
        except:
            iteration_record["start"] = None
            iteration_record["end"] = None

        # Forecast each target column
        for target_col in target_columns:
            model, chunk_pred, err = forecast_missing(history, chunk, target_column=target_col)

            chunk_predictions[target_col] = chunk_pred[[f"{target_col}_predicted"]]
            smape_logs[target_col].append(err)

            iteration_record[f"SMAPE_{target_col}"] = err
            print(f"Iteration {iteration+1} - {target_col} SMAPE: {err:.2f}%")

            chunk[target_col] = chunk_pred[f"{target_col}_predicted"]

        # Save chunk forecast
        forecasts.append(chunk)

        # Update history
        history = pd.concat([history, chunk], ignore_index=True)
        df_pred = df_pred.iloc[steps:]
        iteration += 1

        # Save iteration tracking record
        iteration_tracker.append(iteration_record)

    # Combine all forecasts
    full_forecast_df = pd.concat(forecasts, ignore_index=True)

    # Convert tracker log to DataFrame
    tracker_df = pd.DataFrame(iteration_tracker)

    # Save tracker log to CSV
    tracker_df.to_csv("forecast_iteration_tracker.csv", index=False)

    # Plot SMAPE per target
    plt.figure(figsize=(12, 6))
    for col in target_columns:
        plt.plot(range(1, len(smape_logs[col]) + 1), smape_logs[col], marker='o', label=col)
    plt.title("Recursive Iteration SMAPE per Target Column")
    plt.xlabel("Iteration")
    plt.ylabel("SMAPE (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\n✅ Recursive forecasting complete.")
    for col in target_columns:
        print(f"Average SMAPE for {col}: {np.mean(smape_logs[col]):.2f}%")

    return full_forecast_df, smape_logs, tracker_df

forecast_df, smape_logs, tracker_df = rolling_forecast_multi(
    df_train,
    df_pred,
    target_columns=columns_to_predict,
    steps=48
)

# Save results
forecast_df.to_csv("recursive_forecast_all_targets.csv", index=False)
tracker_df.to_csv("forecast_iteration_tracker.csv", index=False)




