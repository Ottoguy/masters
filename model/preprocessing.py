from load_dans import all_data as data
import pandas as pd
import os
from datetime import datetime
import numpy as np

def export_csv_for_id(df, id_to_export, parent_folder="prints"):
    # If the desired ID is "all", export all rows
    if id_to_export.lower() == "all":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "all")
    # If the desired ID is "meta", export the meta_df
    elif id_to_export.lower() == "meta":
        desired_rows = meta_df.copy()
        output_folder = os.path.join(parent_folder, "meta")
    else:
        # Filter DataFrame based on the desired ID
        desired_rows = df[df['ID'] == id_to_export]
        output_folder = os.path.join(parent_folder, str(id_to_export))

    # Create a folder named "prints" if it doesn't exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Create a subfolder with the ID or "all" as its name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the file name
    output_file = f"{output_folder}/{current_datetime}.csv"

    # Print desired_rows to a CSV file
    desired_rows.to_csv(output_file, index=False)

    if id_to_export.lower() == "meta":
        print(f"Meta exported to: {output_file}")
    elif id_to_export.lower() == "all":
        print(f"All rows exported to: {output_file}")
    else:
        print(f"Exported rows for ID {id_to_export} to: {output_file}")

print("Merging data...")
merged_data = []
for i in enumerate(data):
    for j in enumerate(i[1]):
        merged_data.append(j[1])

columns = ['Filename', 'Timestamp', 'Status', 'Phase1Current', 'Phase2Current', 'Phase3Current',
           'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Value7', 'ChargingStatus', 'ID', 'Value10']

print("Creating dataframe...")
df = pd.DataFrame(merged_data, columns=columns)

# Drop rows with ID 0
df = df[df['ID'] != 0]
# Sort by ID and Timestamp
df = df.sort_values(by=['ID', 'Timestamp'])
# Reset index
df = df.reset_index(drop=True)

# Filter DataFrame based on the desired ID
# desired_id = "59449840"
# desired_rows = df[df['ID'] == desired_id]
# print(desired_rows)

print("Creating meta dataframe...")
# Create a new DataFrame with one row for each unique ID and a 'Half_Minutes' column
meta_df = df.groupby('ID').size().reset_index(name='Half_Minutes')

# Drop the first row in meta_df
meta_df = meta_df.iloc[1:]

# Add new columns for 'Charging' and 'Connected'
charging_counts = df[df['ChargingStatus'] == 'Charging'].groupby(
    'ID').size().reset_index(name='Charging_Half_Minutes')

# Merge the new columns into meta_df
meta_df = pd.merge(meta_df, charging_counts, on='ID', how='left')

# Fill NaN values with 0
meta_df = meta_df.fillna(0)

# Convert 'Charging_Half_Minutes' to integers
meta_df['Charging_Half_Minutes'] = meta_df['Charging_Half_Minutes'].astype(int)

# Assuming 'Half_Minutes' is a column in the 'meta_df' DataFrame
df['Half_Minutes'] = meta_df['Half_Minutes']

# Add new column 'FullyCharged', considering the last contiguous streak of "Charging" values
last_status = df.groupby('ID')['ChargingStatus'].last().reset_index(name='LastStatus')

# Get the last contiguous streak size for each ID
last_streak_size = df[df['ChargingStatus'] == 'Charging'].groupby('ID').size().reset_index(name='LastStreakSize')
print(last_streak_size)

# Merge the last status and last streak size into the meta_df
meta_df = pd.merge(meta_df, last_status, on='ID', how='left')
meta_df = pd.merge(meta_df, last_streak_size, on='ID', how='left')

# Calculate the percentage of the last contiguous streak size
meta_df['StreakPercentage'] = meta_df['LastStreakSize'] / meta_df['Half_Minutes']

# Determine if the car is fully charged based on the conditions
meta_df['FullyCharged'] = (meta_df['LastStatus'] == 'Connected') | (meta_df['StreakPercentage'] < 0.2)

# Drop unnecessary columns
meta_df.drop(columns=['LastStatus', 'LastStreakSize', 'StreakPercentage'], inplace=True)

# Add new columns for 'Time connected' and 'Time disconnected'
first_timestamps = df.groupby(
    'ID')['Timestamp'].first().reset_index(name='TimeConnected')
last_timestamps = df.groupby('ID')['Timestamp'].last(
).reset_index(name='TimeDisconnected')

# Merge the new columns into meta_df
meta_df = pd.merge(meta_df, first_timestamps, on='ID', how='left')
meta_df = pd.merge(meta_df, last_timestamps, on='ID', how='left')

# Add new column 'FilenameSubstring'
filename_substrings = df.groupby('ID')['Filename'].first(
).str[11:-4].reset_index(name='ChargingPoint')

# Merge the new column into meta_df
meta_df = pd.merge(meta_df, filename_substrings, on='ID', how='left')

# Convert 'Phase1Current', 'Phase2Current', 'Phase3Current' to numeric
df[['Phase1Current', 'Phase2Current', 'Phase3Current']] = df[['Phase1Current',
                                                           'Phase2Current', 'Phase3Current']].apply(pd.to_numeric, errors='coerce')
df[['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']] = df[['Phase1Voltage',
                                                              'Phase2Voltage', 'Phase3Voltage']].apply(pd.to_numeric, errors='coerce')

#Make sure all Current and voltag values are positive
df['Phase1Current'] = df['Phase1Current'].abs()
df['Phase2Current'] = df['Phase2Current'].abs()
df['Phase3Current'] = df['Phase3Current'].abs()
df['Phase1Voltage'] = df['Phase1Voltage'].abs()
df['Phase2Voltage'] = df['Phase2Voltage'].abs()
df['Phase3Voltage'] = df['Phase3Voltage'].abs()

# Calculate accumulated kWh for each row in df
# Assuming each row of current lasts 30 seconds
df['Effect'] = (df['Phase1Current']*df["Phase1Voltage"] +
                         df['Phase2Current']*df["Phase2Voltage"] + df['Phase3Current']*df["Phase3Voltage"]) * 30 / 3600

# Calculate total accumulated kWh for each ID
total_kWh = df.groupby('ID')['Effect'].sum().reset_index(name='kWh')

# Divide the 'kWh' column by 1000 so it's in kWh instead of Wh
total_kWh['kWh'] = total_kWh['kWh'] / 1000

# Merge the new column into meta_df
meta_df = pd.merge(meta_df, total_kWh, on='ID', how='left')

# Fill NaN values with 0
meta_df['kWh'] = meta_df['kWh'].fillna(0)

# Convert 'kWh' to float
meta_df['kWh'] = meta_df['kWh'].astype(float)

# Rename the 'kWh' column to 'kWh Charged'
meta_df.rename(columns={'kWh': 'Energy_Uptake'}, inplace=True)

# Round 'kWh_Charged' to 3 decimals
meta_df['Energy_Uptake'] = meta_df['Energy_Uptake'].round(3)

# Add new column 'Current_Type' based on conditions in df
df['Current_Type'] = '3-Phase'
df.loc[df['Phase1Current'] != 0, 'Current_Type'] = '1-Phase'
df.loc[df['Phase2Current'] != 0, 'Current_Type'] = '1-Phase'
df.loc[df['Phase3Current'] != 0, 'Current_Type'] = '1-Phase'

# Add 'Current_Type' column to meta_df
current_type_column = df.groupby(
    'ID')['Current_Type'].first().reset_index(name='Current_Type')
meta_df = pd.merge(meta_df, current_type_column, on='ID', how='left')
meta_df['Current_Type'] = meta_df['Current_Type'].fillna('Unknown')

# List of holiday dates in the format 'YYYY-MM-DD'
holiday_dates = ["2023-12-24", "2023-12-25", "2023-12-26", "2023-12-31", '2024-01-01', "2024-01-06", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-19", "2024-06-06", "2024-06-22", "2024-11-02", "2024-12-25", "2024-12-26"]

# Function to determine if a date is a weekend or a holiday
def is_weekend_or_holiday(timestamp):
    date_format = "%Y-%m-%d-%H:%M:%S.%f" if '.' in timestamp else "%Y-%m-%d-%H:%M:%S"
    date_obj = datetime.strptime(timestamp, date_format)
    
    # Check if the date is a weekend (Saturday or Sunday)
    is_weekend = date_obj.weekday() >= 5
    
    # Check if the date is a holiday
    date_str = date_obj.strftime("%Y-%m-%d")
    is_holiday = date_str in holiday_dates

    return is_weekend or is_holiday

# Add new column 'Weekend' to meta_df
meta_df['Weekend'] = meta_df['TimeConnected'].apply(is_weekend_or_holiday)

# Convert 'Weekend' to boolean
meta_df['Weekend'] = meta_df['Weekend'].astype(bool)

# Drop in df
df.drop(columns=['Half_Minutes', 'Value10', "Current_Type"], inplace=True)

# Encode hours and minutes as a combined cyclical feature
def encode_cyclical_features(df, column_name):
    df[column_name + '_combined_sin'] = np.sin(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))
    df[column_name + '_combined_cos'] = np.cos(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))

# Apply the encoding to 'TimeConnected' and 'TimeDisconnected'
encode_cyclical_features(meta_df, 'TimeConnected')
encode_cyclical_features(meta_df, 'TimeDisconnected')

# Sort by ID
meta_df = meta_df.sort_values(by=['ID'])
# Reset index
meta_df = meta_df.reset_index(drop=True)

# Example: Export CSV for a specific ID or all rows
desired_id_to_export = "meta"  # Or "all" for all rows, or "meta" for meta_df
export_csv_for_id(df, desired_id_to_export)