import pandas as pd
import numpy as np
from functools import reduce
from datetime import datetime
from pathlib import Path

# Load station CSVs
station_files = {
    'Aasu': '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Aasu_ALL_15min_data_cleaned.csv',
    'Poloa': '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Poloa_ALL_15min_data_cleaned.csv',
    'Afono': '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Afono_ALL_15min_data_cleaned.csv',
    'Vaipito': '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/Vaipito_ALL_15min_data_cleaned.csv',
}
cols_to_drop = ['WS_mph_S_WVT', 'WindDir_D1_WVT', 'WindDir_SD1_WVT', 'RECORD', 'BattV_Avg']
required_columns = ['TIMESTAMP','LAT','LON','PTemp_C_Max','AirTF_Avg','SlrW_Avg','SlrMJ_Tot','RH','Rain_in_Tot']

def convert_to_numeric(df):
    for col in df.columns:
        if col != 'TIMESTAMP':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def circular_weight_components(wd_deg_series, w):
    """
    From first code: Convert wind direction to weighted sin/cos components.
    This is the CORRECT way to handle circular wind direction variables.
    """
    wd = pd.to_numeric(wd_deg_series, errors="coerce")
    wd_rad = np.deg2rad(wd)
    return np.sin(wd_rad) * w, np.cos(wd_rad) * w

# Load and prep station data
renamed_stations = []
stations = {}
for name, path in station_files.items():
    df = pd.read_csv(path)
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    df = convert_to_numeric(df)

    if all(col in df.columns for col in required_columns):
        df_renamed = df[required_columns].copy()
        df_renamed = df_renamed.rename(columns={col: f"{col}_{name}" for col in required_columns if col != 'TIMESTAMP'})
        renamed_stations.append(df_renamed)
        stations[name] = df

combined_df = reduce(lambda left, right: pd.merge(left, right, on='TIMESTAMP', how='outer'), renamed_stations)
combined_df = combined_df.iloc[16:].reset_index(drop=True)

def add_distance_to_target(combined_df, target_station):
    tgt_lat = combined_df[f'LAT_{target_station}'].iloc[75164]
    tgt_lon = combined_df[f'LON_{target_station}'].iloc[75164]
    for col in combined_df.columns:
        if 'LAT_' in col:
            name = col.split('_')[1]
            if name != target_station:
                dist = haversine_distance(
                    combined_df[f'LAT_{name}'], combined_df[f'LON_{name}'],
                    tgt_lat, tgt_lon
                )
                combined_df[f'distance_to_{target_station}_{name}'] = dist
    return combined_df

def apply_idw_weights(combined_df, target_station='Vaipito', power=2):
    station_list = [s for s in ['Poloa', 'Aasu', 'Afono'] if s != target_station]
    base_vars = {col.replace(f"_{s}", "") for col in combined_df.columns for s in station_list if col.endswith(f"_{s}")}

    for var in base_vars:
        for s in station_list:
            c = f"{var}_{s}"
            if c in combined_df.columns:
                d = f'distance_to_{target_station}_{s}'
                w = f'weight_{s}'
                combined_df[w] = 1 / (combined_df[d] ** power + 1e-6)
                combined_df[f'weighted_{var}_{s}'] = combined_df[c] * combined_df[w]
    return combined_df

def process_synoptic_file(path):
    """
    Updated with proper circular interpolation from first code.
    """
    df = pd.read_csv(path, skiprows=[1])
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], utc=True, errors='coerce').dt.tz_localize(None)
    df['TIMESTAMP'] = df['TIMESTAMP'].dt.round('15min')
    df = df[df['TIMESTAMP'].notna()]
    
    lat = df['LAT'].iloc[0] if 'LAT' in df.columns else None
    lon = df['LON'].iloc[0] if 'LON' in df.columns else None
    
    df = df[['TIMESTAMP', 'air_temp_set_1', 'relative_humidity_set_1',
             'wind_speed_set_1', 'wind_direction_set_1', 'Elevation']]
    
    df['wind_direction_set_1'] = pd.to_numeric(df['wind_direction_set_1'], errors='coerce')
    df['wind_speed_set_1'] = pd.to_numeric(df['wind_speed_set_1'], errors='coerce')
    df['air_temp_set_1'] = pd.to_numeric(df['air_temp_set_1'], errors='coerce')
    df['relative_humidity_set_1'] = pd.to_numeric(df['relative_humidity_set_1'], errors='coerce')
    df = df.groupby('TIMESTAMP', as_index=False).mean(numeric_only=True)
    
    temp = df.set_index('TIMESTAMP')
    
    if not isinstance(temp.index, pd.DatetimeIndex):
        raise TypeError("TIMESTAMP index is not DatetimeIndex after grouping.")
    
    # ===========================
    #   WIND DIRECTION (circular) - FROM FIRST CODE
    # ===========================
    wd_rad = np.deg2rad(temp['wind_direction_set_1'])
    
    # Unit vectors weighted by wind speed
    u = np.sin(wd_rad)
    v = np.cos(wd_rad)
    
    u_i = temp['wind_speed_set_1'] * u
    v_i = temp['wind_speed_set_1'] * v
    
    try:
        u_interp = u_i.resample('15min').interpolate('akima')
        v_interp = v_i.resample('15min').interpolate('akima')
        
        # Back to direction
        wd_interp = np.rad2deg(np.arctan2(u_interp, v_interp))
        wd_interp = (wd_interp + 360) % 360
    except Exception:
        # Fallback interpolation
        wd_interp = temp['wind_direction_set_1'].resample('15min').interpolate('nearest')
    
    # ===========================
    #   SCALAR VARIABLES
    # ===========================
    df_int = temp[['air_temp_set_1','relative_humidity_set_1',
                   'wind_speed_set_1','Elevation']].resample('15min').interpolate('akima')
    
    # ===========================
    #   COMBINE RESULTS
    # ===========================
    out = df_int.reset_index()
    out['wind_direction_set_1'] = wd_interp.values
    out['LAT'] = lat
    out['LON'] = lon
    
    return out


def integrate_synoptic(df, station_df, station_name, synoptic_dfs):
    """
    Updated to use circular_weight_components from first code for wind direction.
    """
    start, end = station_df['TIMESTAMP'].min(), station_df['TIMESTAMP'].max()
    station_lat = station_df['LAT'].iloc[75164]
    station_lon = station_df['LON'].iloc[75164]

    for i, syn_df in enumerate(synoptic_dfs):
        if not syn_df.empty:
            syn_df = syn_df[(syn_df['TIMESTAMP'] >= start) & (syn_df['TIMESTAMP'] <= end)].copy()
            syn_df = convert_to_numeric(syn_df)
            merged = pd.merge(df[['TIMESTAMP']], syn_df, on='TIMESTAMP', how='left')
            dist = max(haversine_distance(syn_df['LAT'].iloc[0], syn_df['LON'].iloc[0], station_lat, station_lon), 0.1)
            weight = 1 / (dist ** 2)

            # ---- CORRECTED: Use circular components for wind direction ----
            sin_w, cos_w = circular_weight_components(merged['wind_direction_set_1'], weight)
            df[f'wind_direction_sin_weighted_{i}'] = sin_w
            df[f'wind_direction_cos_weighted_{i}'] = cos_w
            
            # ---- Scalar variables (same as before) ----
            df[f'wind_speed_weighted_{i}'] = merged['wind_speed_set_1'] * weight
            df[f'air_temp_weighted_{i}'] = merged['air_temp_set_1'] * weight
            df[f'relative_humidity_weighted_{i}'] = merged['relative_humidity_set_1'] * weight
            df[f'synoptic_elevation_{i}'] = syn_df['Elevation'].iloc[0]
    return df

# Load synoptic files
synoptic_files = [
    '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/NSTU.2022-12-31.csv',
    '/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Raw Data/SFGP6.2022-12-31.csv'
]
synoptic_dfs = [process_synoptic_file(p) for p in synoptic_files]

# Uncomment these based on what is the target, dropping variables that are missing during modeling
#combined_df = combined_df.drop(columns=['RH_Aasu'], errors='ignore')
combined_df = combined_df.drop(columns=['AirTF_Avg_Aasu'], errors='ignore')
#combined_df = combined_df.drop(columns=['SlrMJ_Tot_Poloa'], errors='ignore')
#combined_df = combined_df.drop(columns=['SlrW_Avg_Poloa'], errors='ignore')
#combined_df = combined_df.drop(columns=['SlrW_Avg_Vaipito'], errors='ignore')
#combined_df = combined_df.drop(columns=['SlrMJ_Tot_Vaipito'], errors='ignore')
combined_df = combined_df.drop(columns=['Rain_in_Tot_Aasu'], errors='ignore')

# Example: combine, add distance, IDW, synoptic for target station
target_station = 'Aasu'
combined_df = add_distance_to_target(combined_df, target_station)
combined_df = apply_idw_weights(combined_df, target_station)
combined_df = integrate_synoptic(combined_df, stations[target_station], target_station, synoptic_dfs)

def drop_unused_columns(combined_df):
    drop_cols = [col for col in combined_df.columns if 'distance' in col or 'weight_' in col or 'LON' in col or 'LAT' in col]
    combined_df = combined_df.drop(columns=drop_cols)
    return combined_df

combined_df = drop_unused_columns(combined_df)

date_config = {
    ('Vaipito', 'SlrW_Avg'): {
        "cutoff": "2022-03-23 10:30:00",
        "start": "2022-03-23 10:30:00",
        "end": "2022-08-14 13:30:00"
    },
    ('Vaipito', 'SlrMJ_Tot'): {
        "cutoff": "2022-03-23 10:30:00",
        "start": "2022-03-23 10:30:00",
        "end": "2022-08-14 13:30:00"
    },
    ('Aasu','RH'): {
        "cutoff": "2020-03-09 08:45:00",
        "start": "2020-03-09 08:45:00",
        "end": "2021-04-10 22:00:00"
    },
    ('Aasu','AirTF_Avg'): {
        "cutoff": "2020-03-09 08:45:00",
        "start": "2020-03-09 08:45:00",
        "end": "2021-04-10 22:00:00"
    },
    ('Poloa','SlrW_Avg'): {
        "cutoff": "2022-03-22 06:00:00",
        "start": "2022-03-22 06:00:00",
        "end": "2022-08-14 11:30:00"
    },
    ('Poloa','SlrMJ_Tot'): {
        "cutoff": "2022-03-22 06:00:00",
        "start": "2022-03-22 06:00:00",
        "end": "2022-08-14 11:30:00"
    },
    ('Aasu','Rain_in_Tot'): {
        "cutoff": "2020-04-14 00:20:00",
        "start": "2020-04-14 00:20:00",
        "end": "2022-03-25 13:45:00"
    }
}

def create_train_pred_splits(df, target_station, target_variable, config_dict, simulate_missing=True):
    key = (target_station, target_variable)
    if key not in config_dict:
        raise ValueError(f"No date config found for key {key}")
    
    date_cfg = config_dict[key]
    cutoff = pd.to_datetime(date_cfg['cutoff'])
    print(cutoff)
    start = pd.to_datetime(date_cfg['start'])
    end = pd.to_datetime(date_cfg['end'])

    target_col = f"{target_variable}_{target_station}"
    non_target_features = [c for c in df.columns if c not in ['TIMESTAMP', target_col]]

    df = df.dropna(subset=non_target_features, how='all')
    df_train = df[df['TIMESTAMP'] <= cutoff].dropna()
    df_pred = df[(df['TIMESTAMP'] > start) & (df['TIMESTAMP'] < end)].copy()

    if simulate_missing and target_col in df_pred.columns:
        df_pred[target_col] = np.nan

    return df_train, df_pred

df_train, df_pred = create_train_pred_splits(
    combined_df,
    target_station='Aasu',
    target_variable='RH',
    config_dict=date_config
)
df_train.to_csv("/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Model Input Data/aasu_rh_train.csv", index=False)
# df_pred.to_csv("/Users/lizamclatchy/Documents/Github/ASPA_HistoricalDataCleaning/ASCDP/Data Cleaning/Cleaned Model Input Data/aasu_Rain_in_Tot_pred.csv", index=False)