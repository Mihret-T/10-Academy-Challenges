import pandas as pd
import matplotlib.pyplot as plt
import sys

# Testing
print('Hello World')

# Upload the dataset
data = pd.read_csv("./data/benin-malanville.csv") 
# sierraleone_data = pd.read_csv("./data/sierraleone-bumbuna.csv")
# togo_data = pd.read_csv("./data/togo-dapaong_qc.csv")

# Check if the DataFrame has been loaded properly:
if data is not None and not data.empty:
    print('File Uploaded properly')
else:
    print('Failed to upload the file or file is empty')

# ********************** Summary Statistics *******************************
# Calculate the mean, median, standard deviation,and other statistical measures for each numeric column to understand data distribution.
summary_stats = data.describe()
print("Summary Statistics:\n", summary_stats)
mean_values = data.mean(skipna=True, numeric_only=True)
median_values = data.median(skipna=True, numeric_only=True)
std_deviation = data.std(skipna=True, numeric_only=True)
range_values = data.max(skipna=True, numeric_only=True) - data.min(skipna=True, numeric_only=True)

# Print results
print("\nMean:\n", mean_values)
print("\nMedian:\n", median_values)
print("\nStandard Deviation:\n", std_deviation)
print("\nRange:\n", std_deviation)

# ********************** Data Quality Check *******************************
# Identify missing values
missing_values = data.isnull().sum()
# print("Missing Values:\n", missing_values)

# Check for negative or out-of-range values in GHI, DNI, DHI
ghi_invalid = data[data['GHI'] < 0]
dni_invalid = data[data['DNI'] < 0]
dhi_invalid = data[data['DHI'] < 0]

# print("\nInvalid GHI Entries:\n", ghi_invalid)
# print("\nInvalid DNI Entries:\n", dni_invalid)
# print("\nInvalid DHI Entries:\n", dhi_invalid)

# Outliers 
def detect_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column < lower_bound) | (column > upper_bound)]

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

# ********************** Time Series Analysis *******************************
# Convert the 'timestamp' column to datetime if not already done
copy_data = data.copy()
# Convert 'Timestamp' column to datetime
copy_data['Timestamp'] = pd.to_datetime(copy_data['Timestamp'])
copy_data.set_index('Timestamp', inplace=True)

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

sys.exit()