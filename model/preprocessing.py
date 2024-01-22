from load_dans import all_data as data
import pandas as pd

print("Merging data...")
merged_data = []
for i in enumerate(data):
    for j in enumerate(i[1]):
        merged_data.append(j[1])

columns = ['Filename', 'Timestamp', 'Status', 'Phase1Effect', 'Phase2Effect', 'Phase3Effect',
           'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Value7', 'ChargingStatus', 'ID', 'Value10']

print("Creating dataframe...")
df = pd.DataFrame(merged_data, columns=columns)
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
meta_df.rename(columns={'kWh': 'kWh Charged'}, inplace=True)

# Sort meta_df by 'kWh Charged' in descending order
meta_df = meta_df.sort_values(by='kWh Charged', ascending=False)

print(meta_df)
