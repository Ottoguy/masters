from load_dans import all_data as data
import pandas as pd
import os
from datetime import datetime

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

columns = ['Filename', 'Timestamp', 'Status', 'Phase1Effect', 'Phase2Effect', 'Phase3Effect',
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
# Create a new DataFrame with one row for each unique ID and a 'Rows' column
meta_df = df.groupby('ID').size().reset_index(name='Rows')

# Drop the first row in meta_df
meta_df = meta_df.iloc[1:]

# Add new columns for 'Charging' and 'Connected'
charging_counts = df[df['ChargingStatus'] == 'Charging'].groupby(
    'ID').size().reset_index(name='ChargingCount')
connected_counts = df[df['ChargingStatus'] == 'Connected'].groupby(
    'ID').size().reset_index(name='ConnectedCount')

# Merge the new columns into meta_df
meta_df = pd.merge(meta_df, charging_counts, on='ID', how='left')
meta_df = pd.merge(meta_df, connected_counts, on='ID', how='left')

# Fill NaN values with 0
meta_df = meta_df.fillna(0)

# Convert 'ChargingCount' and 'ConnectedCount' to integers
meta_df['ChargingCount'] = meta_df['ChargingCount'].astype(int)
meta_df['ConnectedCount'] = meta_df['ConnectedCount'].astype(int)

# Add new column 'FullyCharged', indicating whether or not the car was (presumably) fully charged when disconnected
last_status = df.groupby(
    'ID')['ChargingStatus'].last().reset_index(name='LastStatus')
meta_df = pd.merge(meta_df, last_status, on='ID', how='left')
meta_df['FullyCharged'] = meta_df['LastStatus'] == 'Connected'
meta_df.drop(columns='LastStatus', inplace=True)

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

# Convert 'Phase1Effect', 'Phase2Effect', 'Phase3Effect' to numeric
df[['Phase1Effect', 'Phase2Effect', 'Phase3Effect']] = df[['Phase1Effect',
                                                           'Phase2Effect', 'Phase3Effect']].apply(pd.to_numeric, errors='coerce')

# Calculate accumulated kWh for each row in df
# Assuming each row of effects lasts 30 seconds
df['Accumulated_kWh'] = (df['Phase1Effect'] +
                         df['Phase2Effect'] + df['Phase3Effect']) * 30 / 3600

# Calculate total accumulated kWh for each ID
total_kWh = df.groupby('ID')['Accumulated_kWh'].sum().reset_index(name='kWh')

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
df.loc[df['Phase1Effect'] != 0, 'Current_Type'] = '1-Phase'
df.loc[df['Phase2Effect'] != 0, 'Current_Type'] = '1-Phase'
df.loc[df['Phase3Effect'] != 0, 'Current_Type'] = '1-Phase'

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
meta_df['Weekend_Disconnected'] = meta_df['TimeDisconnected'].apply(is_weekend_or_holiday)
meta_df['Weekend_Connected'] = meta_df['TimeConnected'].apply(is_weekend_or_holiday)

# Convert 'Weekend' to boolean
meta_df['Weekend_Disconnected'] = meta_df['Weekend_Disconnected'].astype(bool)
meta_df['Weekend_Connected'] = meta_df['Weekend_Connected'].astype(bool)

# Example: Export CSV for a specific ID or all rows
desired_id_to_export = "meta"  # Or "all" for all rows, or "meta" for meta_df
export_csv_for_id(df, desired_id_to_export)