import os
import glob
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans, KernelKMeans, KShape, silhouette_score
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import numpy as np

# num_cores: Use joblib for parallel processing, set to -1 to use all available cores
# use_all_3_phase_data: Set to True to use all 3 phases for 3-Phase data, set to False to use only Phase1Voltage and Phase1Current
# use_voltage: Set to True to use voltage data, set to False to use only current data
# min_cluster_size: Minimum number of samples in a cluster
# max_cluster_size: Maximum number of samples in a cluster
# handle_min_clusters: Set to 'merge' to merge clusters with less than min_cluster_size samples, set to 'reassign' to reassign samples to other clusters, set to 'outliers' to mark clusters as outliers
# handle_max_clusters: Set to 'split' to split clusters with more than max_cluster_size samples, set to 'reassign' to reassign samples to other clusters
# calculate_silhouette: Set to True to calculate silhouette score, set to False to skip calculation
# algorithm: Set to 'tskmeans' to use TimeSeriesKMeans, set to 'kernelkmeans' to use KernelKMeans, set to 'kshape' to use KShape
# max_iter: Maximum number of iterations for the algorithm
# tol: Tolerance for convergence
# n_init: Number of initializations to perform
# metric: Metric to use for clustering
# max_iter_barycenter: Maximum number of iterations for the barycenter computation
# ts_samples: Number of time series samples to use
def TsClustering(ts_samples, num_clusters, algorithm, max_iter, tol, n_init, metric, max_iter_barycenter, use_voltage, use_all3_phases, min_cluster_size, max_cluster_size, handle_min_clusters, handle_max_clusters, calculate_silhouette):
    # Specify the directory where your files are located
    folder_path = 'prints/extracted/' + str(ts_samples) + '/'

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

    cluster_df = pd.DataFrame(columns=['ID', 'Cluster'])
    cluster_df['ID'] = df['ID'].unique()

    cluster_df['NumClusters'] = num_clusters
    cluster_df['Algorithm'] = algorithm
    cluster_df['MaxIter'] = max_iter
    cluster_df['Tolerance'] = tol
    cluster_df['NInit'] = n_init
    cluster_df['Metric'] = metric
    cluster_df['MaxIterBarycenter'] = max_iter_barycenter
    cluster_df['UseVoltage'] = use_voltage
    cluster_df['UseAll3Phases'] = use_all3_phases
    cluster_df['MinClusterSize'] = min_cluster_size
    cluster_df['MaxClusterSize'] = max_cluster_size
    cluster_df['HandleMinClusters'] = handle_min_clusters
    cluster_df['HandleMaxClusters'] = handle_max_clusters

    print('Data loaded from:', latest_file)
    print('Scaling Time Series Clustering for:', algorithm)
    scaler_voltage = StandardScaler()
    scaler_current = StandardScaler()

    df['Phase1Current'] = scaler_current.fit_transform(df['Phase1Current'].values.reshape(-1, 1))
    if use_all3_phases:
        df['Phase2Current'] = scaler_current.fit_transform(df['Phase2Current'].values.reshape(-1, 1))
        df['Phase3Current'] = scaler_current.fit_transform(df['Phase3Current'].values.reshape(-1, 1))
    if use_voltage:
        df['Phase1Voltage'] = scaler_voltage.fit_transform(df['Phase1Voltage'].values.reshape(-1, 1))
        if use_all3_phases:
            df['Phase2Voltage'] = scaler_voltage.fit_transform(df['Phase2Voltage'].values.reshape(-1, 1))
            df['Phase3Voltage'] = scaler_voltage.fit_transform(df['Phase3Voltage'].values.reshape(-1, 1))

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

    X = to_time_series_dataset(df_list)
    labels = []

    def TSKMeans(labels, cluster_df, num_clusters, max_iter, tol, n_init, metric, max_iter_barycenter):
        print('Clustering with', num_clusters, 'clusters')
        model = TimeSeriesKMeans(n_clusters=num_clusters, metric=metric, max_iter=max_iter, tol=tol, n_init=n_init, max_iter_barycenter=max_iter_barycenter, verbose=True, random_state=42, n_jobs=-1)
        labels = model.fit_predict(X)
        return labels, cluster_df

    def KKMeans(labels, cluster_df, num_clusters, max_iter, tol, n_init):
        print('Clustering with', num_clusters, 'clusters')
        model = KernelKMeans(n_clusters=num_clusters, max_iter=max_iter, tol=tol, n_init=n_init, kernel="gak", verbose=True, random_state=42, n_jobs=-1)
        labels = model.fit_predict(X)
        return labels, cluster_df

    def KS(labels, cluster_df, num_clusters, max_iter, tol, n_init):
        print('Clustering with', num_clusters, 'clusters')
        model = KShape(n_clusters=num_clusters, max_iter=max_iter, tol=tol, n_init=n_init, verbose=True, random_state=42)
        labels = model.fit_predict(X)
        return labels, cluster_df

    if algorithm == 'tskmeans':
        print('Clustering with TimeSeriesKMeans')
        labels, cluster_df = TSKMeans(labels, cluster_df, num_clusters, max_iter, tol, n_init, metric, max_iter_barycenter)
    elif algorithm == 'kernelkmeans':
        print('Clustering with KernelKMeans')
        labels, cluster_df = KKMeans(labels, cluster_df, num_clusters, max_iter, tol, n_init)
    elif algorithm == 'kshape':
        print('Clustering with KShape')
        labels, cluster_df = KS(labels, cluster_df, num_clusters, max_iter, tol, n_init)

    # Add cluster labels to cluster_df
    cluster_df['Cluster'] = labels

    print('Clustering done, handling min and max clusters now...')

    print('Clustering done, handling min and max clusters now...')
    print('Unique labels:', np.unique(labels))
    print('len of labels:', len(labels))
    print('len of cluster_df:', len(cluster_df))
    print('len of X:', len(X))
    print('Unique ids in cluster_df:', cluster_df['ID'].nunique())
    #Heads of X an cluster_df and labels
    print('X:', X[:5])
    print('cluster_df:', cluster_df.head())
    print('labels:', labels[:5])

    unique_labels, counts = np.unique(labels, return_counts=True)
    min_clusters_to_handle = unique_labels[counts < min_cluster_size]
    print('Clusters with less than the limit ', min_cluster_size, 'samples:', min_clusters_to_handle)
    max_clusters_to_handle = unique_labels[counts > max_cluster_size]
    print('Clusters with more than the limit ', max_cluster_size, 'samples:', max_clusters_to_handle)

    #Handling min clusters
    if min_clusters_to_handle.size > 0:
        if handle_min_clusters == 'merge':
            print('Merging clusters with less than the limit ', min_cluster_size, 'samples')
            for cluster in min_clusters_to_handle:
                # Find the centroid of the cluster
                centroid = X[labels == cluster].mean(axis=0)
                # Calculate distances between centroid and centroids of other clusters
                distances = [np.linalg.norm(centroid - X[labels == lbl].mean(axis=0)) for lbl in unique_labels if lbl != cluster]
                # Find the index of the closest cluster
                closest_cluster_index = np.argmin(distances)
                closest_cluster_label = unique_labels[closest_cluster_index]
                
                # Update labels for samples in the current cluster
                cluster_indices = np.where(labels == cluster)[0]
                labels[cluster_indices] = closest_cluster_label
        elif handle_min_clusters == 'reassign':
            print('Reassigning clusters with less than the limit ', min_cluster_size, 'samples')
            for cluster in min_clusters_to_handle:
                # Get indices of samples in the current cluster
                cluster_indices = np.where(labels == cluster)[0]
                for index in cluster_indices:
                    print('Reassigning sample', index)
                    # Get the current sample
                    sample = X[index]
                    print('Sample:', sample)
                    # Calculate distances between the sample and all centroids
                    distances = [np.linalg.norm(sample - centroid) for centroid in X[labels != cluster].mean(axis=0)]
                    print('Distances:', distances)
                    # Find the index of the nearest centroid
                    nearest_centroid_index = np.argmin(distances)
                    print('Nearest centroid index:', nearest_centroid_index)
                    # Get the label of the nearest cluster
                    nearest_cluster_label = (labels[labels != cluster])[nearest_centroid_index]
                    # Assign the sample to the nearest cluster
                    labels[index] = nearest_cluster_label
        elif handle_min_clusters == 'outliers':
            print('Marking clusters with less than the limit ', min_cluster_size, 'samples as outliers')
            for cluster in min_clusters_to_handle:
                cluster_df['Cluster'].replace({cluster: -1}, inplace=True)
                
    #Handling max clusters
    if max_clusters_to_handle.size > 0:
        if handle_max_clusters == 'split':
            print('Splitting clusters with more than the limit ', max_cluster_size, 'samples')
            for cluster in max_clusters_to_handle:
                # Initialize two new clusters
                new_cluster1_label = max(unique_labels) + 1
                new_cluster2_label = max(unique_labels) + 2
                unique_labels = np.append(unique_labels, [new_cluster1_label, new_cluster2_label])
                
                # Find the centroid of the cluster
                centroid = X[labels == cluster].mean(axis=0)
                
                # Identify points in the cluster
                cluster_indices = np.where(labels == cluster)[0]
                
                # Calculate distances between centroid and all points in the cluster
                distances = [np.linalg.norm(centroid - X[index]) for index in cluster_indices]
                
                # Sort indices of points by their distances to the centroid
                sorted_indices = np.array(cluster_indices)[np.argsort(distances)]
                
                # Distribute points from the cluster to the new clusters
                for i, index in enumerate(sorted_indices):
                    if i % 2 == 0:
                        labels[index] = new_cluster1_label
                    else:
                        labels[index] = new_cluster2_label

        elif handle_max_clusters == 'reassign':
            print('Reassigning clusters with more than the limit ', max_cluster_size, 'samples')
            for cluster in max_clusters_to_handle:
                # Find the centroid of the cluster
                centroid = X[labels == cluster].mean(axis=0)
                # Get indices of points in the current cluster
                cluster_indices = np.where(labels == cluster)[0]
                # Sort cluster points by distance to centroid in descending order
                sorted_indices = sorted(cluster_indices, key=lambda idx: np.linalg.norm(X[idx] - centroid), reverse=True)
                
                for index in sorted_indices:
                    # Calculate distances between the current point and all centroids except the centroid of the current cluster
                    distances = [np.linalg.norm(X[index] - centroid) for centroid in X[labels != cluster].mean(axis=0)]
                    # Find the index of the nearest centroid
                    nearest_centroid_index = np.argmin(distances)
                    # Get the label of the nearest cluster
                    nearest_cluster_label = (labels[labels != cluster])[nearest_centroid_index]
                    # Assign the point to the nearest cluster
                    labels[index] = nearest_cluster_label
                    # Check if the size of the current cluster is within the desired range
                    if len(X[labels == cluster]) <= max_cluster_size:
                        break

    # Save the figure with the current date and time in the filename
    results_dir = "prints/ts_clustering/" + str(ts_samples) + "/" + str(num_clusters) + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Update labels to cluster_df
    cluster_df['Cluster'] = labels

    print('Handling done.')
    print('Unique labels:', np.unique(labels))
    print('len of labels:', len(labels))
    print('len of cluster_df:', len(cluster_df))
    print('len of X:', len(X))
    print('Unique ids in cluster_df:', cluster_df['ID'].nunique())
    #Heads of X an cluster_df and labels
    print('X:', X[:5])
    print('cluster_df:', cluster_df.head())
    print('labels:', labels[:5])

    if calculate_silhouette:
        # Calculate silhouette score
        print('Calculating silhouette score')
        silhouette = silhouette_score(X, labels, metric=metric, n_jobs=-1)
    else:
        #Create a list of None values with the same length as if the silhouette score was calculated
        silhouette = [None] * len(cluster_df)

    # Append silhouette score to cluster_df
    cluster_df['SilhouetteScore'] = silhouette

    # Save the DataFrame to a CSV file
    output_file = os.path.join(results_dir, f"clustering_results_{current_datetime}.csv")
    cluster_df.to_csv(output_file, index=False)

    print("Results saved to:", output_file)

    #Make a df "return_df" with the first row of cluster_df
    return_df = cluster_df.iloc[[0]].copy()
    #Drop the columns 'ID' and 'Cluster' from the DataFrame
    return_df.drop(columns=['ID', 'Cluster'], inplace=True)
    #Reorder so 'SilhouetteScore' is the first column
    return_df = return_df[['SilhouetteScore'] + [col for col in return_df.columns if col != 'SilhouetteScore']]

    eval_folder = 'prints/ts_eval'
    # Create a folder if it doesn't exist
    if not os.path.exists(eval_folder):
        os.makedirs(eval_folder)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the file name
    print(f"Creating the file {current_datetime}.csv")
    output_file = f"{eval_folder}/{current_datetime}.csv"
    # Print desired_rows to a CSV file
    return_df.to_csv(output_file, index=False)
    #Print path to the created file
    print(f"Results saved to {output_file}")

    return return_df