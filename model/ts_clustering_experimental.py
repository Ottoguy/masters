import os
import glob
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Function to reshape a dataframe into a time series dataset
def reshape_to_time_series(df, use_all_3_phase_data=False):
    if use_all_3_phase_data:
        time_series_columns = ['Phase1Voltage', 'Phase1Current', 'Phase2Voltage', 'Phase3Voltage', 'Phase2Current', 'Phase3Current']
    else:
        time_series_columns = ['Phase1Voltage', 'Phase1Current']

    time_series_list = []

    for _, group in df.groupby('ID'):
        features = group[time_series_columns].values
        time_series_list.append(features)

    time_series_dataset = to_time_series_dataset(time_series_list)
    return time_series_dataset

# num_cores: Use joblib for parallel processing, set to -1 to use all available cores
def TsClusteringExperimental(num_cores, num_clusters, distance_metric, ts_samples, use_current_type, use_all_3_phase_data,
                 Use_time, use_charging_point, use_floor, use_weekend, use_maxvoltage, use_maxcurrent, use_energy_uptake,
                 use_average_voltage, use_average_current):
    
    # Specify the directory where your files are located
    folder_path = 'prints/extracted/' + str(ts_samples) + '/'
    meta_path = 'prints/meta/'

    # Create a pattern to match files in the specified format
    file_pattern = '*'
    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)
    # Take the latest file
    latest_file = file_list[0]
    # Load your data from the latest file
    df = pd.read_csv(latest_file)

    meta_list = glob.glob(os.path.join(meta_path, file_pattern))
    meta_list.sort(key=os.path.getmtime, reverse=True)
    latest_meta = meta_list[0]
    meta_df = pd.read_csv(latest_meta)

    data = df[['ID', 'Phase1Voltage', 'Phase1Current']].copy()
    categorical_features = []
    # Columns to ignore during iteration
    ignore_columns = []
    #ignore_columns = ['ID', 'Phase1Voltage', 'Phase1Current']

    if use_current_type:
        data = data.join(meta_df['Current_Type'])
        categorical_features.append('Current_Type')
    if use_all_3_phase_data:
        data = data.join(df[['Phase2Voltage', 'Phase3Voltage', 'Phase2Current', 'Phase3Current']])
        #ignore_columns.extend(['Phase2Voltage', 'Phase3Voltage', 'Phase2Current', 'Phase3Current'])
    if Use_time:
        data = data.join(meta_df['TimeConnected'])
        categorical_features.append('TimeConnected')
    if use_charging_point:
        data = data.join(meta_df['ChargingPoint'])
        categorical_features.append('ChargingPoint')
    if use_floor:
        data = data.join(meta_df['Floor'])
        categorical_features.append('Floor')
    if use_weekend:
        data = data.join(meta_df['Weekend'])
        categorical_features.append('Weekend')
    if use_maxvoltage:
        data = data.join(meta_df['MaxVoltage'])
        categorical_features.append('MaxVoltage')
    if use_maxcurrent:
        data = data.join(meta_df['MaxCurrent'])
        categorical_features.append('MaxCurrent')
    if use_energy_uptake:
        data = data.join(meta_df['Energy_Uptake'])
        categorical_features.append('Energy_Uptake')
    if use_average_voltage:
        data = data.join(meta_df['AverageVoltageDifference'])
        categorical_features.append('AverageVoltageDifference')
    if use_average_current:
        data = data.join(meta_df['AverageCurrentDifference'])
        categorical_features.append('AverageCurrentDifference')

    print("Scaling time series data...")
    # Separate standard scaling for 'Voltage' and 'Current'
    scaler_voltage = StandardScaler()
    scaler_current = StandardScaler()

    # Fit and transform each feature
    data['Phase1Voltage'] = scaler_voltage.fit_transform(data[['Phase1Voltage']].dropna())
    data['Phase1Current'] = scaler_current.fit_transform(data[['Phase1Current']].dropna())

    if use_all_3_phase_data:
        data['Phase2Voltage'] = scaler_voltage.fit_transform(data[['Phase2Voltage']].dropna())
        data['Phase3Voltage'] = scaler_voltage.fit_transform(data[['Phase3Voltage']].dropna())
        data['Phase2Current'] = scaler_current.fit_transform(data[['Phase2Current']].dropna())
        data['Phase3Current'] = scaler_current.fit_transform(data[['Phase3Current']].dropna())

    # Round the values to 3 decimals
    data = data.round({'Phase1Voltage': 3, 'Phase1Current': 3})
    if use_all_3_phase_data:
        data = data.round({'Phase2Voltage': 3, 'Phase3Voltage': 3, 'Phase2Current': 3, 'Phase3Current': 3})

    # Get all columns except for ignore_columns
    relevant_columns = [col for col in data.columns if col not in ignore_columns]

    # Iterate through unique combinations of all features
    unique_combinations = data.groupby(categorical_features, as_index=False)[relevant_columns]

    for name, group in unique_combinations:
         # Rename the dataframe based on the feature values
        df_name = '_'.join([f"{feature}_{value}" for feature, value in zip(categorical_features, name)])
        
        # Save the dataframe with a unique name
        globals()[df_name] = group

    # Print the number of created dataframes
    print(f"Number of created dataframes: {len([var for var in globals() if isinstance(globals()[var], pd.DataFrame)])}")

    # Print the names of the created dataframes
    print("Names of created dataframes:")
    for var in globals():
        if isinstance(globals()[var], pd.DataFrame):
            print(var)

   # Reshape each dataframe into a time series dataset
    for var in globals():
        if isinstance(globals()[var], pd.DataFrame):
            print(f"Reshaping {var} into a time series dataset...")
            print(f"Shape of {var}: {globals()[var].shape}")
            print(f"{var}:\n{globals()[var]}")
            time_series_dataset = reshape_to_time_series(globals()[var], use_all_3_phase_data)

    # Save the figure with the current date and time in the filename
    results_dir = "prints/ts_clustering/" + str(ts_samples) + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Iterate over different numbers of clusters
    for num_clusters in num_clusters_1_phase_range:
        print(f"Clustering time series data with {num_clusters} clusters...")

        # Apply TimeSeriesKMeans clustering with DTW as the metric
        kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric=distance_metric, n_jobs=num_cores, verbose=True)
        labels = kmeans.fit_predict(time_series_1_phase_dataset)

        # Calculate silhouette score
        s_score = silhouette_score(time_series_1_phase_dataset, labels, metric=distance_metric, n_jobs=num_cores)

        # Print silhouette score for the current number of clusters
        print(f"Silhouette Score for {num_clusters} clusters: {s_score}")

        # Create a subfolder for the current number of clusters
        subfolder_path = os.path.join(results_dir, f"1-Phase_{num_clusters}_clusters")
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Save the clustered data to a file within the subfolder
        clustered_data_file_path = 'clustered_data_'
        clustered_data = pd.DataFrame(
            {'ID': time_series_data_1_phase['ID'].unique(), 'Cluster': labels, 'Silhouette Score': s_score, 'Num Clusters': num_clusters})
        clustered_data.to_csv(os.path.join(subfolder_path, clustered_data_file_path + current_datetime + "ts_samples" + str(ts_samples) +
                                        "distance_metric" + distance_metric + "num_clusters" + str(num_clusters) + ").csv"), index=False)

    for num_clusters in num_clusters_3_phase_range:
        print(f"Clustering time series data with {num_clusters} clusters...")

        # Apply TimeSeriesKMeans clustering with DTW as the metric
        kmeans = TimeSeriesKMeans(n_clusters=num_clusters, metric=distance_metric, n_jobs=num_cores, verbose=True)
        labels = kmeans.fit_predict(time_series_3_phase_dataset)

        # Calculate silhouette score
        s_score = silhouette_score(time_series_3_phase_dataset, labels, metric=distance_metric, n_jobs=num_cores)

        # Print silhouette score for the current number of clusters
        print(f"Silhouette Score for {num_clusters} clusters: {s_score}")

        # Create a subfolder for the current number of clusters
        subfolder_path = os.path.join(results_dir, f"3-Phase_{num_clusters}_clusters")
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

        # Save the clustered data to a file within the subfolder
        clustered_data_file_path = 'clustered_data_'
        clustered_data = pd.DataFrame(
            {'ID': time_series_data_3_phase['ID'].unique(), 'Cluster': labels, 'Silhouette Score': s_score, 'Num Clusters': num_clusters})
        clustered_data.to_csv(os.path.join(subfolder_path, clustered_data_file_path + current_datetime + " (use_all_3_phases"
                                        + str(use_all_3_phase_data) + "ts_samples" + str(ts_samples) + "distance_metric" + distance_metric +
                                        "num_clusters" + str(num_clusters) + ").csv"), index=False)