import os
import glob
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# num_cores: Use joblib for parallel processing, set to -1 to use all available cores
# num_clusters_1_phase_range: Range of numbers of clusters to try for 1-Phase data
# num_clusters_3_phase_range: Range of numbers of clusters to try for 3-Phase data
# use_all_3_phase_data: Set to True to use all 3 phases for 3-Phase data, set to False to use only Phase1Voltage and Phase1Current
def TsClustering(num_cores, num_clusters_1_phase_range, num_clusters_3_phase_range, use_all_3_phase_data, distance_metric, ts_samples):
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

    # Extract relevant columns
    time_series_data_1_phase = df[['ID', 'Phase1Voltage', 'Phase1Current']].copy()
    if use_all_3_phase_data:
        time_series_data_3_phase = df[['ID', 'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Phase1Current', 'Phase2Current', 'Phase3Current']].copy()
    else:
        time_series_data_3_phase = df[['ID', 'Phase1Voltage', 'Phase1Current']].copy()

    print("Scaling time series data...")
    # Separate standard scaling for 'Phase1Voltage' and 'Phase1Current'
    scaler_voltage = StandardScaler()
    scaler_current = StandardScaler()

    # Fit and transform each feature
    time_series_data_1_phase['Phase1Voltage'] = scaler_voltage.fit_transform(time_series_data_1_phase[['Phase1Voltage']].dropna())
    time_series_data_1_phase['Phase1Current'] = scaler_current.fit_transform(time_series_data_1_phase[['Phase1Current']].dropna())
    time_series_data_3_phase['Phase1Voltage'] = scaler_voltage.fit_transform(time_series_data_3_phase[['Phase1Voltage']].dropna())
    time_series_data_3_phase['Phase1Current'] = scaler_current.fit_transform(time_series_data_3_phase[['Phase1Current']].dropna())
    if use_all_3_phase_data:
        time_series_data_3_phase['Phase2Voltage'] = scaler_voltage.fit_transform(time_series_data_3_phase[['Phase2Voltage']].dropna())
        time_series_data_3_phase['Phase3Voltage'] = scaler_voltage.fit_transform(time_series_data_3_phase[['Phase3Voltage']].dropna())
        time_series_data_3_phase['Phase2Current'] = scaler_current.fit_transform(time_series_data_3_phase[['Phase2Current']].dropna())
        time_series_data_3_phase['Phase3Current'] = scaler_current.fit_transform(time_series_data_3_phase[['Phase3Current']].dropna())

    # Round the values to 3 decimals
    time_series_data_1_phase = time_series_data_1_phase.round({'Phase1Voltage': 3, 'Phase1Current': 3})
    if use_all_3_phase_data:
        time_series_data_3_phase = time_series_data_3_phase.round({'Phase1Voltage': 3, 'Phase2Voltage': 3, 'Phase3Voltage': 3, 'Phase1Current': 3, 'Phase2Current': 3, 'Phase3Current': 3})
    else:
        time_series_data_3_phase = time_series_data_3_phase.round({'Phase1Voltage': 3, 'Phase1Current': 3})

    #Divide the time series data into one dataframe for Current_Type in meta_df = "1-Phase" and = "3-Phase"
    time_series_data_1_phase = time_series_data_1_phase[time_series_data_1_phase['ID'].isin(meta_df[meta_df['Current_Type'] == "1-Phase"]['ID'])]
    time_series_data_3_phase = time_series_data_3_phase[time_series_data_3_phase['ID'].isin(meta_df[meta_df['Current_Type'] == "3-Phase"]['ID'])]

    # Reshape the DataFrame with variable length time series
    time_series_1_phase_list = []
    time_series_3_phase_list = []

    for id_value, group in time_series_data_1_phase.groupby('ID'):
        features = group[['Phase1Voltage', 'Phase1Current']].values
        time_series_1_phase_list.append(features)

    for id_value, group in time_series_data_3_phase.groupby('ID'):
        if use_all_3_phase_data:
            features = group[['Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Phase1Current', 'Phase2Current', 'Phase3Current']].values
        else:
            features = group[['Phase1Voltage', 'Phase1Current']].values
        time_series_3_phase_list.append(features)

    # Convert to time series dataset if needed
    time_series_1_phase_dataset = to_time_series_dataset(time_series_1_phase_list)
    time_series_3_phase_dataset = to_time_series_dataset(time_series_3_phase_list)

    # Print the shape of the reshaped data
    print("Reshaped 1-Phase Data Shape:", time_series_1_phase_dataset.shape)
    print("Reshaped 3-Phase Data Shape:", time_series_3_phase_dataset.shape)

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