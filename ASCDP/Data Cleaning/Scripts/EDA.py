import requests
import pandas as pd
from io import StringIO
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import datetime
from statsmodels.tsa.stattools import adfuller #adfuller method
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# Function to fetch CSV file URLs from a GitHub repository
def get_csv_files_from_repo(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        csv_files = {
            file['name']: file['download_url']
            for file in files
            if file['name'].endswith('.csv') and 'download_url' in file
        }
        return csv_files
    else:
        raise ValueError(f"Failed to fetch repository contents: {response.status_code}")

# Function to read a CSV file from a URL into a DataFrame
def read_csv_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes
    return pd.read_csv(StringIO(response.text))

# GitHub API endpoint for the repository contents
api_url = "https://api.github.com/repos/cshuler/Hydro_Monitoring_Network_ASPA-UH/contents/Scripts/Liza_Aimee_Workspace/Data"

# Get the list of CSV files
csv_files = get_csv_files_from_repo(api_url)

# Check if the list of CSV files is populated
if not csv_files:
    raise ValueError("No CSV files found in the repository")

# Dictionary to store DataFrames with full file names as keys
dataframes = {}

# Read each CSV file and store in the dictionary
for file_name, url in csv_files.items():
    try:
        # Sanitize the file name to use as a key (remove special characters)
        sanitized_name = re.sub(r'\W+', '_', file_name.split('.')[0])
        df = read_csv_from_url(url)
        dataframes[sanitized_name] = df
    except Exception as e:
        print(f"Error reading {file_name}: {e}")

# Processing ALL and Bad dataframes
for bad_name, bad_df in dataframes.items():
    if 'BAD' in bad_name.upper():
        # Match corresponding ALL file
        station_name = bad_name.replace('Bad_data', '').strip('_')
        matching_all_keys = [key for key in dataframes if station_name in key and 'ALL' in key.upper()]
        
        if not matching_all_keys:
            print(f"No matching ALL data for {bad_name}")
            continue
        
        all_name = matching_all_keys[0]  # Assume the first match is correct
        all_df = dataframes[all_name]
        
        # Ensure TIMESTAMP exists in the ALL DataFrame
        if 'TIMESTAMP' not in all_df.columns:
            print(f"Skipping {all_name}: 'TIMESTAMP' column not found")
            continue
        
        # Convert TIMESTAMP to datetime in ALL DataFrame
        all_df['TIMESTAMP'] = pd.to_datetime(all_df['TIMESTAMP'], errors='coerce')
        
        # Convert Bad data start and end to datetime
        bad_df['Bad data Start'] = pd.to_datetime(bad_df['Bad data Start'], errors='coerce')
        bad_df['Bad data End'] = pd.to_datetime(bad_df['Bad data End'], errors='coerce')
        
        # Apply NaN based on Bad data conditions
        for _, row in bad_df.iterrows():
            affected_column = row['Data affected']
            start_date = row['Bad data Start']
            end_date = row['Bad data End']
            
            if pd.notnull(start_date) and pd.notnull(end_date):
                mask = (all_df['TIMESTAMP'] >= start_date) & (all_df['TIMESTAMP'] <= end_date)
                
                if affected_column.upper() == 'ALL':
                    # Set all columns except TIMESTAMP to NaN
                    all_df.loc[mask, all_df.columns.difference(['TIMESTAMP'])] = np.nan
                    print(f"Set all columns to NaN in {all_name} from {start_date} to {end_date}")
                elif affected_column in all_df.columns:
                    # Set specific column to NaN
                    all_df.loc[mask, affected_column] = np.nan
                    print(f"Set {affected_column} to NaN in {all_name} from {start_date} to {end_date}")
                else:
                    print(f"Column {affected_column} not found in {all_name}")



#Test for stationarity
def adf_test_with_timestamp(df, timestamp_col, numeric_col, title=''):
    """
    Perform Augmented Dickey-Fuller Test on a numeric column indexed by a timestamp.

    Args:
        df (pd.DataFrame): DataFrame containing the time-series data.
        timestamp_col (str): Name of the timestamp column.
        numeric_col (str): Name of the numeric column to test.
        title (str): Title for the test output.
    """
    print(f'Augmented Dickey-Fuller Test on "{title}"')
    
    # Ensure timestamp column is a datetime index
    try:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.dropna(subset=[timestamp_col])
        df.set_index(timestamp_col, inplace=True)
    except Exception as e:
        print(f"Error setting timestamp index for {title}: {e}")
        return
    
    # Run the ADF test on the numeric column
    try:
        series = df[numeric_col].dropna()
        result = adfuller(series, autolag='AIC')
        labels = ['ADF test statistic', 'p-value', '# lags used', '# observations']
        out = pd.Series(result[0:4], index=labels)

        for key, val in result[4].items():
            out[f'critical value ({key})'] = val
        print(out.to_string())
        if result[1] <= 0.05:
            print("Strong evidence against the null hypothesis, reject the null, data has no unit root and is stationary")
        else:
            print("Weak evidence against the null hypothesis, time series has a unit root, indicating it is non-stationary\n")
    except Exception as e:
        print(f"Error running ADF test for {numeric_col} in {title}: {e}")


def exploratory_data_analysis_and_adf(dataframes):
    for name, df in dataframes.items():
        # Filter DataFrames whose keys contain 'ALL'
        if 'ALL' in name.upper():
            print(f"Analyzing DataFrame: {name}")
            
            # Ensure TIMESTAMP exists and convert to datetime
            if 'TIMESTAMP' not in df.columns:
                print(f"Skipping {name}: 'TIMESTAMP' column not found")
                continue
            
            # Convert TIMESTAMP to datetime
            try:
                df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
                df = df.dropna(subset=['TIMESTAMP'])
            except Exception as e:
                print(f"Error processing TIMESTAMP for {name}: {e}")
                continue
            
            # Convert all columns to numeric where possible (exclude TIMESTAMP)
            for column in df.columns:
                if column != 'TIMESTAMP':
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    except Exception as e:
                        print(f"Error converting {column} in {name} to numeric: {e}")
            
            # Exclude columns that are non-numeric or explicitly unwanted
            excluded_columns = ['RECORD']
            numeric_columns = [col for col in df.columns if col not in excluded_columns and col != 'TIMESTAMP']
            
            # Check if any valid numeric columns exist
            if not numeric_columns:
                print(f"No numeric columns to analyze for {name}")
                continue
            
            # Perform ADF test and plot each numeric column against TIMESTAMP
            for column in numeric_columns:
                #Plot
                plt.figure(figsize=(10, 6))
                plt.plot(df['TIMESTAMP'], df[column], label=column, marker='o', linestyle='-')
                plt.title(f"{column} vs TIMESTAMP ({name})")
                plt.xlabel("TIMESTAMP")
                plt.ylabel(column)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                
                plt.hist(df[column],bins=40 ,label=column)
                plt.title(f"{column} histogram")
                plt.xlabel(column)
                plt.show()
                
                #ADF Test
                #adf_test_with_timestamp(df.copy(), 'TIMESTAMP', column, title=f"{column} ({name})")
                
   #              column_data = df[column].dropna()  # Drop NaNs
   #              n_lags = min(len(column_data) // 2 - 1, 40)  # Use 40 as a safe upper limit for most cases
   # # Use half the length of the data for lags

   #              # Plot ACF
   #              fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns
            
   #          # Plot ACF
   #              plot_acf(column_data, lags=n_lags, ax=axes[0])
   #              axes[0].set_title(f"Autocorrelation Function (ACF) - {column}")
            
   #          # Plot PACF
   #              plot_pacf(column_data, lags=n_lags, ax=axes[1])
   #              axes[1].set_title(f"Partial Autocorrelation Function (PACF) - {column}")
            
   #          # Adjust layout
   #              plt.tight_layout()
   #              plt.show()
exploratory_data_analysis_and_adf(dataframes)
 



# processed_dataframes = {}

# for name, df in dataframes.items():
#     # Check if the key contains 'ALL' (case-insensitive)
#     if 'ALL' in name.upper():
#         # Filter only TIMESTAMP and Rain_in_Tot columns
#         filtered_df = df[['TIMESTAMP', 'Rain_in_Tot']].copy()
#         processed_dataframes[f"rain_df_{name}"] = filtered_df

# monthly_averages = {}
# for name, df in processed_dataframes.items():
#     # Ensure TIMESTAMP column is in datetime format
#     df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
#     df['Rain_in_Tot'] = pd.to_numeric(df['Rain_in_Tot'], errors='coerce')

#     # Drop rows with NaT in TIMESTAMP or NaN in Rain_in_Tot
#     df = df.dropna(subset=['TIMESTAMP', 'Rain_in_Tot'])
    
#     # Group by year and month, then calculate the mean Rain_in_Tot for each month
#     df['YearMonth'] = df['TIMESTAMP'].dt.to_period('M')  # Create a year-month period column
#     monthly_avg = (
#         df.groupby('YearMonth')['Rain_in_Tot']
#         .mean()
#         .reset_index(name='Monthly_Average')
#     )
    
#     # Store the result in the new dictionary
#     monthly_averages[name] = monthly_avg
                
# full_yearmonth_range = pd.period_range('2017-01', '2022-12', freq='M')

# # Initialize an empty dictionary to store results
# all_data = {}

# for name, df in processed_dataframes.items():
#     # Ensure TIMESTAMP column is in datetime format
#     df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], errors='coerce')
    
#     # Drop rows with NaT in TIMESTAMP
#     df = df.dropna(subset=['TIMESTAMP'])
    
#     # Convert Rain_in_Tot to numeric, coercing errors to NaN
#     df['Rain_in_Tot'] = pd.to_numeric(df['Rain_in_Tot'], errors='coerce')
    
#     # Drop rows with NaN in Rain_in_Tot after conversion
#     df = df.dropna(subset=['Rain_in_Tot'])
    
#     # Group by YearMonth and calculate the average
#     df['YearMonth'] = df['TIMESTAMP'].dt.to_period('M')
#     monthly_avg = df.groupby('YearMonth')['Rain_in_Tot'].sum()
    
#     # Reindex to the full range, filling missing values with NaN
#     monthly_avg = monthly_avg.reindex(full_yearmonth_range, fill_value=np.nan)
    
#     # Store the result in the dictionary
#     all_data[name] = monthly_avg

# # Create a consolidated DataFrame with all results
# consolidated_df = pd.DataFrame(all_data).transpose()

# # Rename columns to string format for clarity
# consolidated_df.columns = [str(col) for col in consolidated_df.columns]

# # Display the resulting DataFrame
# print(consolidated_df)
# yearly_sums = consolidated_df.groupby(lambda col: col[:4], axis=1).sum()

# # Generate x positions for each station
# x = np.arange(len(yearly_sums.index))  # Position of stations on the x-axis
# width = 0.05  # Width of each bar
# years = yearly_sums.columns  # List of years

# # Create the plot
# fig, ax = plt.subplots(figsize=(12, 6))

# # Plot each year's data with proper offsets
# for i, year in enumerate(years):
#     ax.bar(
#         x + i * width, 
#         yearly_sums[year], 
#         width=width, 
#         label=year
#     )

# # Customize the plot
# ax.set_title('Total Rainfall Per Station (Grouped by Year)', fontsize=16)
# ax.set_xlabel('Station', fontsize=14)
# ax.set_ylabel('Rainfall Sums (in)', fontsize=14)
# ax.set_xticks(x + width * (len(years) - 1) / 2)
# ax.set_xticklabels(yearly_sums.index, rotation=45, fontsize=12)
# ax.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.tight_layout()

# # Show the plot
# plt.show()

# total_avg_rainfall_per_station = consolidated_df.mean(axis=1)

