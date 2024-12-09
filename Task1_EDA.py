import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from windrose import WindroseAxes

def summary_stat(data):
    print("\nSummary Statistics:")
    print(data.info())
    summary_stats = data.describe()
    print("Summary Statistics:\n", summary_stats)
    columns = data.columns.tolist()
    print("Column Names:\n", columns)

    # Calculate the mean, median, standard deviation,and other statistical measures 
    mean_values = data.mean(skipna=True, numeric_only=True)
    median_values = data.median(skipna=True, numeric_only=True)
    std_deviation = data.std(skipna=True, numeric_only=True)
    range_values = data.max(skipna=True, numeric_only=True) - data.min(skipna=True, numeric_only=True)
        
    print("\nMean:\n", mean_values)
    print("\nMedian:\n", median_values)
    print("\nStandard Deviation:\n", std_deviation)
    print("\nRange:\n", range_values)
   
def quality_check(data):
    print("\nData Quality Check:")
    # Identify missing values
    missing_values = data.isnull().sum()
    print("Missing Values:\n", missing_values)

    # Check for negative or out-of-range values in GHI, DNI, DHI
    ghi_invalid = data[data['GHI'] < 0]
    dni_invalid = data[data['DNI'] < 0]
    dhi_invalid = data[data['DHI'] < 0]

    print("\nInvalid GHI Entries:\n", ghi_invalid)
    print("\nInvalid DNI Entries:\n", dni_invalid)
    print("\nInvalid DHI Entries:\n", dhi_invalid)
    
    ghi_outliers = detect_outliers(data['GHI'])
    dni_outliers = detect_outliers(data['DNI'])
    dhi_outliers = detect_outliers(data['DHI'])
    
    print("\nOutliers in GHI:\n", ghi_outliers)
    print("\nOutliers in DNI:\n", dni_outliers)
    print("\nOutliers in DHI:\n", dhi_outliers)
    
    # Outliers for sensors and wind speed
    moda_outliers = detect_outliers(data['ModA'])
    modb_outliers = detect_outliers(data['ModB'])
    ws_outliers = detect_outliers(data['WS'])
    wsgust_outliers = detect_outliers(data['WSgust'])

    print("\nOutliers in ModA:\n", moda_outliers)
    print("\nOutliers in ModB:\n", modb_outliers)
    print("\nOutliers in WS:\n", ws_outliers)
    print("\nOutliers in WSgust:\n", wsgust_outliers)

def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

def time_series_analysis(data):
    print("\n Time Series Analysis :")
    # Convert 'Timestamp' column to datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)

    # Plot time series for GHI, DNI, DHI, and Tamb
    plt.figure(figsize=(12, 8))

    # Plot GHI
    plt.subplot(2, 2, 1)
    plt.plot(data['GHI'], label="GHI", color="blue")
    plt.title("Global Horizontal Irradiance (GHI) Over Time")
    plt.xlabel("Time")
    plt.ylabel("GHI")
    plt.legend()

    # Plot DNI
    plt.subplot(2, 2, 2)
    plt.plot(data['DNI'], label="DNI", color="orange")
    plt.title("Direct Normal Irradiance (DNI) Over Time")
    plt.xlabel("Time")
    plt.ylabel("DNI")
    plt.legend()

    # Plot DHI
    plt.subplot(2, 2, 3)
    plt.plot(data['DHI'], label="DHI", color="green")
    plt.title("Diffuse Horizontal Irradiance (DHI) Over Time")
    plt.xlabel("Time")
    plt.ylabel("DHI")
    plt.legend()

    # Plot Tamb
    plt.subplot(2, 2, 4)
    plt.plot(data['Tamb'], label="Tamb", color="red")
    plt.title("Ambient Temperature (Tamb) Over Time")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def correlation_analysis(data):
    print("\n Correlation Analysis :")
     # Select relevant columns for correlation analysis
    columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    correlation_matrix = data[columns].corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix: Solar Radiation and Temperature")
    plt.show()

    # Scatter Matrix for Wind Conditions and Solar Irradiance
    wind_solar_columns = ['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'WD']
    scatter_data = data[wind_solar_columns].dropna()
    plt.figure(figsize=(12, 8))
    scatter_matrix(scatter_data, figsize=(12, 8), alpha=0.7, diagonal='hist')
    plt.suptitle("Scatter Matrix: Wind Conditions and Solar Irradiance")
    
    plt.tight_layout()
    plt.show()

def wind_analysis(data):
    print("\n Wind Analysis :")     
    # Remove missing or infinite values in wind data
    wind_data = data[['WS', 'WD']].dropna()
    wind_data = wind_data[~wind_data.isin([float('inf'), float('-inf')]).any(axis=1)]
    wind_data = wind_data[(wind_data['WD'] >= 0) & (wind_data['WD'] <= 360)]
    # Create wind rose plot
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)

    # Create the wind rose using wind speed and direction
    ax.bar(wind_data['WD'], wind_data['WS'], bins=16, edgecolor='black')

    ax.set_title("Wind Rose: Wind Speed and Direction Distribution")
    ax.set_legend(title="Wind Speed (m/s)")
    plt.show()

def temp_analysis(data):
    print("\n Temperature Analysis :")     
    temp_rh_data = data[['RH', 'Tamb', 'TModA', 'TModB', 'GHI', 'DNI', 'DHI']].dropna()

    # Calculate correlation matrix
    correlation_matrix = temp_rh_data.corr()
    # Plot a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix: RH, Temperature, and Solar Radiation")
    plt.show()

def histogram(data):
    print("\n Histogram :")  
    # Selecting the relevant columns and dropping missing values
    hist_data = data[['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'TModA', 'TModB']].dropna()
    variables = ['GHI', 'DNI', 'DHI', 'WS', 'Tamb', 'TModA', 'TModB']
    colors = ['blue', 'green', 'orange', 'purple', 'red', 'cyan', 'magenta']

    plt.figure(figsize=(16, 12))

    # Plot each variable in a loop
    for i, var in enumerate(variables):
        plt.subplot(3, 3, i + 1)  # Create a grid of subplots
        plt.hist(hist_data[var], bins=30, color=colors[i], alpha=0.7, edgecolor='black')
        plt.title(f'Histogram of {var}')
        plt.xlabel(var)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def Z_score_analysis(data):
    print("\n Z-Score Analysis :") 
    # Select numeric columns for Z-score calculation
    numeric_data = data.select_dtypes(include=[np.number])

    # Calculate Z-scores
    z_scores = (numeric_data - numeric_data.mean()) / numeric_data.std()

    # Add Z-scores to the dataset for review (optional)
    data_with_zscores = pd.concat([data, z_scores.add_suffix('_zscore')], axis=1)

    # Boxplot to visualize outliers
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=z_scores)
    plt.title("Z-Score Distribution")
    plt.ylabel("Z-Score")
    plt.xticks(rotation=45)
    plt.show()

def bubble_analysis(data):
    print("\n Bubble Analysis :") 
    # Select relevant columns and drop missing values
    bubble_data = data[['GHI', 'Tamb', 'WS', 'RH', 'BP']].dropna()
    # Assign variables for axes and bubble size
    x = bubble_data['GHI']  # X-axis
    y = bubble_data['Tamb']  # Y-axis
    bubble_size = bubble_data['RH']  # Bubble size (e.g., Relative Humidity)
    color = bubble_data['WS']  # Bubble color (e.g., Wind Speed)

    plt.figure(figsize=(12, 8))

    # Bubble chart
    scatter = plt.scatter(x, y, s=bubble_size * 10, c=color, cmap='viridis', alpha=0.7, edgecolor='black')
    # Add a color bar for wind speed
    cbar = plt.colorbar(scatter)
    cbar.set_label('Wind Speed (WS)')

    plt.xlabel('GHI (Global Horizontal Irradiance)')
    plt.ylabel('Tamb (Ambient Temperature)')
    plt.title('Bubble Chart: GHI vs Tamb vs WS')
    plt.grid(True)

    plt.show()

def data_cleaning(data):
    # Check for missing values in each column
    missing_summary = data.isnull().sum()
    print("Missing Values:\n", missing_summary)

    # Drop entirely null columns
    data = data.dropna(axis=1, how='all')
    print("Remaining Columns after dropping nulls:\n", data.columns)

data = pd.read_csv("./data/benin-malanville.csv") 
# Check if the DataFrame has been loaded properly:
if data is not None and not data.empty:
    print('File Uploaded properly')
    summary_stat(data)
    # quality_check(data)
    # time_series_analysis(data)
    # correlation_analysis(data)
    # wind_analysis(data)
    # temp_analysis(data)
    # histogram(data)
    # Z_score_analysis(data)
    # bubble_analysis(data)
    # data_cleaning(data)
else:
    print('Failed to upload the file or file is empty')
    sys.exit()
  
