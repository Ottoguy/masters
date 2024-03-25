import os
import glob
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

# num_cores: Use joblib for parallel processing, set to -1 to use all available cores
# num_clusters_1_phase_range: Range of numbers of clusters to try for 1-Phase data
# num_clusters_3_phase_range: Range of numbers of clusters to try for 3-Phase data
# use_all_3_phase_data: Set to True to use all 3 phases for 3-Phase data, set to False to use only Phase1Voltage and Phase1Current
def TsClusteringExperimental(num_cores, ts_samples, algorithm, max_iter, tol, metric, max_iter_barycenter, use_voltage, use_all3_phases, min_cluster_size, max_cluster_size):
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

    print('Data loaded from:', latest_file)
    print('Scaling Time Series Clustering for:', algorithm)
    scaler_voltage = StandardScaler()
    scaler_current = StandardScaler()

    # Scale the data
    df['Phase1Current'] = scaler_current.fit_transform(df['Phase1Current'].values.reshape(-1, 1))
    if use_all3_phases:
        df['Phase2Current'] = scaler_current.fit_transform(df['Phase2Current'].values.reshape(-1, 1))
        df['Phase3Current'] = scaler_current.fit_transform(df['Phase3Current'].values.reshape(-1, 1))
    if use_voltage:
        df['Phase1Voltage'] = scaler_voltage.fit_transform(df['Phase1Voltage'].values.reshape(-1, 1))
        if use_all3_phases:
            df['Phase2Voltage'] = scaler_voltage.fit_transform(df['Phase2Voltage'].values.reshape(-1, 1))
            df['Phase3Voltage'] = scaler_voltage.fit_transform(df['Phase3Voltage'].values.reshape(-1, 1))

    #Convert to float32
    df['Phase1Current'] = df['Phase1Current'].astype(np.float32)
    if use_all3_phases:
        df['Phase2Current'] = df['Phase2Current'].astype(np.float32)
        df['Phase3Current'] = df['Phase3Current'].astype(np.float32)
    if use_voltage:
        df['Phase1Voltage'] = df['Phase1Voltage'].astype(np.float32)
        if use_all3_phases:
            df['Phase2Voltage'] = df['Phase2Voltage'].astype(np.float32)
            df['Phase3Voltage'] = df['Phase3Voltage'].astype(np.float32)

    df_list = []

    for id_value, group in df.groupby('ID'):
        if use_voltage and use_all3_phases:
            features = group[['Phase1Voltage', 'Phase1Current', 'Phase2Voltage', 'Phase2Current', 'Phase3Voltage', 'Phase3Current']].values
        elif use_voltage:
            features = group[['Phase1Voltage', 'Phase1Current']].values
        elif use_all3_phases:
            features = group[['Phase1Current', 'Phase2Current', 'Phase3Current']].values
        else:
            features = group[['Phase1Current']].values
        df_list.append(features)

    
    # Save the figure with the current date and time in the filename
    results_dir = "prints/ts_clustering/" + str(ts_samples) + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
