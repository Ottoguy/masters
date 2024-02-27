import pandas as pd
from datetime import datetime
from functions import encode_cyclical_features
from functions import export_csv_for_id
import numpy as np

# Preprocessing function
def preprocessing(data, ts_samples, meta_lower_bound, empty_charge, streak_percentage, should_filter_1911001328A_2_and_1911001328A_1, export_meta, export_extracted, export_filtered, export_all, export_specific_id, id_to_export, strict_charge_extract):

    def load_data(data):
        print("Merging data...")
        merged_data = []
        for i in enumerate(data):
            for j in enumerate(i[1]):
                merged_data.append(j[1])

        columns = ['Filename', 'Timestamp', 'Status', 'Phase1Current', 'Phase2Current', 'Phase3Current',
                'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Value7', 'ChargingStatus', 'ID', 'Value10']

        print("Creating dataframe...")
        df = pd.DataFrame(merged_data, columns=columns)
        # Sort by ID and Timestamp
        df = df.sort_values(by=['ID', 'Timestamp'])
        # Reset index
        df = df.reset_index(drop=True)
        # Drop rows with ID 0
        df = df[df['ID'] != 0]
        df.drop(columns=['Value7', 'Value10'], inplace=True)

        return df

    def create_meta(df):
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

        return meta_df

    def filter_meta(meta_df, meta_lower_bound):
        print("Filtering meta dataframe...")
        # Filter out IDs with 'Half_Minutes' below meta_lower_bound
        initial_count = len(meta_df)
        meta_df = meta_df[meta_df['Half_Minutes'] >= meta_lower_bound]
        filtered_half_minutes = initial_count - len(meta_df)
        print(f"Filtered {filtered_half_minutes} IDs from meta_df with Half_Minutes below {meta_lower_bound}.")

        return meta_df

    def time_connected_disconnected(df, meta_df):
        # Add new columns for 'Time connected' and 'Time disconnected'
        first_timestamps = df.groupby(
            'ID')['Timestamp'].first().reset_index(name='TimeConnected')
        last_timestamps = df.groupby('ID')['Timestamp'].last(
        ).reset_index(name='TimeDisconnected')
        # Merge the new columns into meta_df
        meta_df = pd.merge(meta_df, first_timestamps, on='ID', how='left')
        meta_df = pd.merge(meta_df, last_timestamps, on='ID', how='left')
        
        return meta_df

    def max_voltage(df, meta_df):
        # Group by 'ID' in df and find the maximum value of 'Phase1Voltage', 'Phase2Voltage', and 'Phase3Voltage'
        max_voltages = df.groupby('ID')[['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']].max()

        # Merge the result back to meta_df based on 'ID'
        meta_df = pd.merge(meta_df, max_voltages, on='ID', how='left')

        # Calculate the overall maximum voltage and create a new column 'MaxVoltage'
        meta_df['MaxVoltage'] = meta_df[['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']].max(axis=1)

        # Drop the unnecessary columns from meta_df
        meta_df.drop(['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage'], axis=1, inplace=True)
        return meta_df

    def max_current(df, meta_df):
        # Group by 'ID' in df and find the maximum value of 'Phase1Current', 'Phase2Current', and 'Phase3Current'
        max_currents = df.groupby('ID')[['Phase1Current', 'Phase2Current', 'Phase3Current']].max()

        # Merge the result back to meta_df based on 'ID'
        meta_df = pd.merge(meta_df, max_currents, on='ID', how='left')

        # Calculate the overall maximum current and create a new column 'MaxCurrent'
        meta_df['MaxCurrent'] = meta_df[['Phase1Current', 'Phase2Current', 'Phase3Current']].max(axis=1)

        # Drop the unnecessary columns from meta_df
        meta_df.drop(['Phase1Current', 'Phase2Current', 'Phase3Current'], axis=1, inplace=True)
        return meta_df

    def filter_zero_values(meta_df):
        # Count the rows before filtering
        initial_row_count = len(meta_df)

        #Convert MaxVoltage and MaxCurrent to float
        meta_df['MaxVoltage'] = meta_df['MaxVoltage'].astype(float)
        meta_df['MaxCurrent'] = meta_df['MaxCurrent'].astype(float)

        # Filter away rows where either 'MaxVoltage' or 'MaxCurrent' is equal to 0
        filtered_df = meta_df[(meta_df['MaxVoltage'] != 0) & (meta_df['MaxCurrent'] != 0)]

        # Count the rows after filtering
        final_row_count = len(filtered_df)

        # Print the count of filtered rows
        print(f"Filtered {initial_row_count - final_row_count} ids in meta_df with either MaxVoltage or MaxCurrent equal to 0.")

        return filtered_df

    #Function for sorting away all rows with IDs that are not in meta_df
    def filter_df(df, meta_df):
        filtered_df = df[df['ID'].isin(meta_df['ID'])]
        print(f"Filtered {len(df) - len(filtered_df)} rows in df whose ids have already been filtered from meta_df.")
        return filtered_df

    #Filter away df strictly
    def charge60(df, meta_df, empty_charge):
        #Filter away all IDs where none of the first empty_charge rows in df are charging
        ids_to_filter = df[df['ChargingStatus'] == 'Charging'].groupby('ID').size().reset_index(name='Charging_Half_Minutes')
        ids_to_filter = ids_to_filter[ids_to_filter['Charging_Half_Minutes'] < empty_charge]
        filtered_df = df[~df['ID'].isin(ids_to_filter['ID'])]
        filtered_meta_df = meta_df[~meta_df['ID'].isin(ids_to_filter['ID'])]
        print(f"Filtered {len(df) - len(filtered_df)} rows based on the first {empty_charge} rows (if not any charging).")
        return filtered_df, filtered_meta_df

    #Filter away ids in meta_df that are not in df
    def filter_meta_df(df, meta_df):
        filtered_meta_df = meta_df[meta_df['ID'].isin(df['ID'])]
        print(f"Filtered {len(meta_df) - len(filtered_meta_df)} ids in meta_df whose ids have already been filtered from df.")
        return filtered_meta_df

    def fully_charged(df, meta_df, streak_percentage):
        # Add new column 'FullyCharged', considering the last contiguous streak of "Charging" values
        last_status = df.groupby('ID')['ChargingStatus'].last().reset_index(name='LastStatus')
        # Get the last contiguous streak size for each ID
        last_streak_size = df[df['ChargingStatus'] == 'Charging'].groupby('ID').size().reset_index(name='LastStreakSize')
        # Merge the last status and last streak size into the meta_df
        meta_df = pd.merge(meta_df, last_status, on='ID', how='left')
        meta_df = pd.merge(meta_df, last_streak_size, on='ID', how='left')
        # Calculate the percentage of the last contiguous streak size
        meta_df['StreakPercentage'] = meta_df['LastStreakSize'] / meta_df['Half_Minutes']
        # Determine if the car is fully charged based on the conditions
        meta_df['FullyCharged'] = (meta_df['LastStatus'] == 'Connected') | (meta_df['StreakPercentage'] < streak_percentage)
        # Drop unnecessary columns
        meta_df.drop(columns=['LastStatus', 'LastStreakSize', 'StreakPercentage'], inplace=True)

        return meta_df

    def charging_point(df, meta_df):
        # Add new column 'FilenameSubstring'
        filename_substrings = df.groupby('ID')['Filename'].first().str[11:-4].reset_index(name='ChargingPoint')
        # Merge the new column into meta_df
        meta_df = pd.merge(meta_df, filename_substrings, on='ID', how='left')
        df = pd.merge(df, filename_substrings, on='ID', how='left')

        return df, meta_df

    #Exclude all rows with chargingpoint=1911001328A_2 or 1911001328A_1
    def filter_chargingpoint(df, meta_df, should_filter_1911001328A_2_and_1911001328A_1):
        if should_filter_1911001328A_2_and_1911001328A_1:
            filtered_df = df[~df['ChargingPoint'].str.contains('1911001328A_1|1911001328A_2')]
            print(f"Filtered {len(df) - len(filtered_df)} rows in df with ChargingPoint 1911001328A_1 or 1911001328A_2.")
            filtered_meta_df = meta_df[~meta_df['ChargingPoint'].str.contains('1911001328A_1|1911001328A_2')]
            print(f"Filtered {len(meta_df) - len(filtered_meta_df)} ids in meta_df with ChargingPoint 1911001328A_1 or 1911001328A_2.")
            return filtered_df, filtered_meta_df
        else:
            return df, meta_df

    def effect(df):
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
        return df

    def energy_uptake(df, meta_df):
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
        return meta_df

    def current_type(df, meta_df):
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
        return meta_df

    # Function to determine if a date is a weekend or a holiday
    def is_weekend_or_holiday(timestamp):
        # List of holiday dates in the format 'YYYY-MM-DD'
        holiday_dates = ["2023-12-24", "2023-12-25", "2023-12-26", "2023-12-31", '2024-01-01', "2024-01-06", "2024-03-29", "2024-04-01", "2024-05-01", "2024-05-09", "2024-05-19", "2024-06-06", "2024-06-22", "2024-11-02", "2024-12-25", "2024-12-26"]

        date_format = "%Y-%m-%d-%H:%M:%S.%f" if '.' in timestamp else "%Y-%m-%d-%H:%M:%S"
        date_obj = datetime.strptime(timestamp, date_format)
        
        # Check if the date is a weekend (Saturday or Sunday)
        is_weekend = date_obj.weekday() >= 5
        
        # Check if the date is a holiday
        date_str = date_obj.strftime("%Y-%m-%d")
        is_holiday = date_str in holiday_dates

        return is_weekend or is_holiday

    def weekend(meta_df):
        # Add new column 'Weekend' to meta_df
        meta_df['Weekend'] = meta_df['TimeConnected'].apply(is_weekend_or_holiday)

        # Convert 'Weekend' to boolean
        meta_df['Weekend'] = meta_df['Weekend'].astype(bool)
        return meta_df

    def cyclical_time(df, meta_df):
        # Convert temporary columns to datetime format for encoding cyclical features
        meta_df['TimeConnected_temp'] = pd.to_datetime(meta_df['TimeConnected'], format='%Y-%m-%d-%H:%M:%S.%f')
        meta_df['TimeDisconnected_temp'] = pd.to_datetime(meta_df['TimeDisconnected'], format='%Y-%m-%d-%H:%M:%S.%f')

        # Shuffling names
        meta_df.rename(columns={'TimeConnected': 'TimeConnected_true'}, inplace=True)
        meta_df.rename(columns={'TimeDisconnected': 'TimeDisconnected_true'}, inplace=True)
        meta_df.rename(columns={'TimeConnected_temp': 'TimeConnected'}, inplace=True)
        meta_df.rename(columns={'TimeDisconnected_temp': 'TimeDisconnected'}, inplace=True)

        # Apply the encoding to 'TimeConnected_temp' and 'TimeDisconnected_temp'
        encode_cyclical_features(meta_df, 'TimeConnected')
        encode_cyclical_features(meta_df, 'TimeDisconnected')

        #Same, for df
        df["Timestamp_temp"] = df["Timestamp"]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d-%H:%M:%S.%f')
        encode_cyclical_features(df, 'Timestamp')
        df['Timestamp'] = df["Timestamp_temp"]
        df.drop(columns=['Timestamp_temp'], inplace=True)
        df.drop(columns=['Status'], inplace=True)

        # Drop the temporary columns
        meta_df.drop(columns=['TimeConnected', 'TimeDisconnected'], inplace=True)

        #Give back the original names
        meta_df.rename(columns={'TimeConnected_true': 'TimeConnected'}, inplace=True)
        meta_df.rename(columns={'TimeDisconnected_true': 'TimeDisconnected'}, inplace=True)
        
        return df, meta_df

    def calculate_average_voltage_difference(df, meta_df, placeholder_value=0.0):
        # Filter rows where at least one voltage value is nonzero
        filtered_df = df[(df['Phase1Voltage'] != 0) | (df['Phase2Voltage'] != 0) | (df['Phase3Voltage'] != 0)]

        # Group by ID and calculate the total difference for each group
        grouped_df = filtered_df.groupby('ID').apply(lambda group: group[['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']].diff().abs().sum(axis=1).sum())

        # Calculate the average for each group normalized by the number of rows
        average_voltage_difference = grouped_df.groupby('ID').mean() / filtered_df.groupby('ID').size()

        # Add the result as a new column to meta_df with placeholder value for IDs with no rows
        meta_df['AverageVoltageDifference'] = meta_df['ID'].map(average_voltage_difference).fillna(placeholder_value)

        return meta_df

    def calculate_average_current_difference(df, meta_df, placeholder_value=0.0):
        # Filter rows where at least one current value is nonzero
        filtered_df = df[(df['Phase1Current'] != 0) | (df['Phase2Current'] != 0) | (df['Phase3Current'] != 0)]

        # Group by ID and calculate the total difference for each group
        grouped_df = filtered_df.groupby('ID').apply(lambda group: group[['Phase1Current', 'Phase2Current', 'Phase3Current']].diff().abs().sum(axis=1).sum())

        # Calculate the average for each group normalized by the number of rows
        average_current_difference = grouped_df.groupby('ID').mean() / filtered_df.groupby('ID').size()

        # Add the result as a new column to meta_df with placeholder value for IDs with no rows
        meta_df['AverageCurrentDifference'] = meta_df['ID'].map(average_current_difference).fillna(placeholder_value)

        return meta_df

    def voltage_diff(df):
        # Add 'VoltageDiff' column
        print("Adding VoltageDiff column...")
        voltage_columns = ['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']
        df['VoltageDiff'] = df[voltage_columns].apply(lambda row: np.max(row) - np.min(row), axis=1)
        return df

    def current_diff(df):
        # Add 'CurrentDiff' column
        print("Adding CurrentDiff column...")
        current_columns = ['Phase1Current', 'Phase2Current', 'Phase3Current']
        df['CurrentDiff'] = df[current_columns].apply(lambda row: np.max(row) - np.min(row), axis=1)
        return df

    #Edit the original dataframe to only include rows that are not in meta_df
    def discarded_df(original, meta_df):
        filtered_df = original[~original['ID'].isin(meta_df['ID'])]
        print(f"Separated {len(original) - len(filtered_df)} rows in original df to discarded folder whose ids have already been filtered from meta_df.")
        return filtered_df

    def extract(df, ts_samples, strict_charge_extract):
        # Extract ts_samples timestamps for each ID
        extracted_df = df.groupby('ID').head(ts_samples)
        if strict_charge_extract:
            #Remove all rows with IDs whose last row is not Charging
            ids_to_remove = extracted_df[extracted_df['ChargingStatus'] != 'Charging']['ID']
            extracted_df = extracted_df[~extracted_df['ID'].isin(ids_to_remove)]
        return extracted_df

    # Load the data
    df = load_data(data)
    # Save Original df
    original = df.copy()
    # Create the meta dataframe
    meta_df = create_meta(df)
    # Filter the meta dataframe
    meta_df = filter_meta(meta_df, meta_lower_bound)
    # Add columns for 'Time connected' and 'Time disconnected' to meta_df
    meta_df = time_connected_disconnected(df, meta_df)
    # Add new column 'MaxVoltage' to meta_df
    meta_df = max_voltage(df, meta_df)
    # Add new column 'MaxCurrent' to meta_df
    meta_df = max_current(df, meta_df)
    # Filter away rows where either 'MaxVoltage' or 'MaxCurrent' is equal to 0
    meta_df = filter_zero_values(meta_df)
    # Filter df based on meta_df
    df = filter_df(df, meta_df)
    # Filter df based on first 60 rows
    df, meta_df = charge60(df, meta_df, empty_charge)
    # Filter meta_df based on df
    meta_df = filter_meta_df(df, meta_df)
    # Determine if the car is fully charged, streak_percentage is how many percent of the last contiguous streak of "Charging" (as a part of all "Charging"-values) for the car to be considered fully charged
    meta_df = fully_charged(df, meta_df, streak_percentage)
    # Add new column 'ChargingPoint' to df and meta_df
    df, meta_df = charging_point(df, meta_df)
    # Filter df based on ChargingPoint
    df, meta_df = filter_chargingpoint(df, meta_df, should_filter_1911001328A_2_and_1911001328A_1)
    # Add new column 'Effect' to df, and cleanup Voltage and Current values
    df = effect(df)
    # Add new column 'Energy_Uptake' to meta_df
    meta_df = energy_uptake(df, meta_df)
    # Add new column 'Current_Type' to df and meta_df
    meta_df = current_type(df, meta_df)
    # Add new column 'Weekend' to meta_df
    meta_df = weekend(meta_df)
    # Add new columns for cyclical time features to df and meta_df
    df, meta_df = cyclical_time(df, meta_df)
    # Calculate the average voltage difference and add it as a new column to meta_df
    meta_df = calculate_average_voltage_difference(df, meta_df)
    # Calculate the average current difference and add it as a new column to meta_df
    meta_df = calculate_average_current_difference(df, meta_df)
    # Add new column 'VoltageDiff' to df
    df = voltage_diff(df)
    # Add new column 'CurrentDiff' to df
    df = current_diff(df)
    # Filtered df
    filtered_df = discarded_df(original, df)
    #Make new dataframe with ts_samples extracted timestamps from each ID
    extracted_df = extract(df, ts_samples, strict_charge_extract)

    if export_meta:
        export_csv_for_id(meta_df, "meta")
    if export_extracted:
        export_csv_for_id(extracted_df, "extracted")
    if export_filtered:
        export_csv_for_id(filtered_df, "filtered")
    if export_all:
        export_csv_for_id(df, "all")
    if export_specific_id:
        export_csv_for_id(df, id_to_export)

#data = DansMästaren data
#ts_samples = number of timestamps to extract for each ID for Time Series Clustering
#meta_lower_bound = the lower bound for the number of half minutes for an ID to be included in the meta dataframe
#empty_charge = the number of half minutes in the beginning of a time series for an ID to be considered as an empty charge and discarded
#streak_percentage = the percentage of the last contiguous streak of "Charging" values for the car to be considered fully charged
#should_filter_1911001328A_2_and_1911001328A_1 = whether to filter away all rows with chargingpoint=1911001328A_2 or 1911001328A_1
#export_meta = whether to export the meta dataframe to a csv file
#export_extracted = whether to export the extracted dataframe to a csv file
#export_filtered = whether to export the filtered dataframe to a csv file (data that has been ifltered away from the main dataframe)
#export_all = whether to export the original dataframe to a csv file
#export_specific_id = whether to export the original dataframe to a csv file with a specific ID
#id_to_export = the specific ID to export
#strict_charge_extract = whether to remove all rows with IDs whose last row is not Charging from the extracted dataframe
