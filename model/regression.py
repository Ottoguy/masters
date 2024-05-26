import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# num_cores: Use joblib for parallel processing, set to -1 to use all available cores
# ts_samples: Number of time series samples to use
# include_ts_clusters: Set to True to include time series clusters, set to False to exclude time series clusters
# phase: Phase to use for clustering
# clusters: Number of clusters to use for clustering
# test_size: Proportion of the dataset to include in the test split
# random_state: Controls the shuffling applied to the data before applying the split
# n_estimators: The number of trees in the forest
def Regression(num_cores, ts_samples, include_ts_clusters, phase, clusters, test_size, random_state, n_estimators):

    print("Loading data for regression")
    # Specify the directory where your files are located
    folder_immediate_path = 'prints/preproc_immediate/'
    folder_intermediate_path = 'prints/preproc_intermediate/'
    folder_final_path = 'prints/preproc_final/'

    origin = phase + "_" + str(clusters) + "_clusters"
    ts_cluster_folder = 'prints/ts_clustering/' + str(ts_samples) + "/" + origin + '/'
    cluster_file_name = ""

    # Create a pattern to match files in the specified format
    file_pattern = '*'

    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_immediate_path, file_pattern))
    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)
    # Take the latest file
    latest_file = file_list[0]
    # Load your data from the latest file
    df_immediate = pd.read_csv(latest_file)

    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_intermediate_path, file_pattern))
    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)
    # Take the latest file
    latest_file = file_list[0]
    # Load your data from the latest file
    df_intermediate = pd.read_csv(latest_file)

    #One-hot encode the 'Current_Type' column
    df_intermediate = pd.get_dummies(df_intermediate, columns=['Current_Type'])

    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_final_path, file_pattern))
    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)
    # Take the latest file
    latest_file = file_list[0]
    # Load your data from the latest file
    df_final = pd.read_csv(latest_file)

    if include_ts_clusters:
        # Get a list of all files in the specified format within the chosen subfolder
        id_cluster_files = glob.glob(os.path.join(ts_cluster_folder, '*.csv'))
        # Sort the files based on modification time (latest first)
        id_cluster_files.sort(key=os.path.getmtime, reverse=True)
        # Take the latest file from the chosen subfolder
        latest_id_cluster_file = id_cluster_files[0]
        # Load your ID-cluster mapping data from the latest file
        df_clusters = pd.read_csv(latest_id_cluster_file)
        #Replace NaN values with -1 (some ID's have not been clustered?)
        df_clusters['Cluster'] = df_clusters['Cluster'].fillna(-1)

        #save name of cluster file to string
        cluster_file_name = latest_id_cluster_file.split("/")[-1]
        cluster_file_name = cluster_file_name.split(".")[0]

        #Remove all rows from the other dataframes with ids that are not in the cluster dataframe
        df_immediate = df_immediate[df_immediate['ID'].isin(df_clusters['ID'])]
        df_intermediate = df_intermediate[df_intermediate['ID'].isin(df_clusters['ID'])]
        df_final = df_final[df_final['ID'].isin(df_clusters['ID'])]
        
        #Print how many rows were removed
        print("Removed", len(df_immediate) - len(df_clusters), "rows from the immediate dataframe (filtered in preprocessing or because of phase?)")

    # List of features to normalize
    features_to_normalize = ['MaxVoltage','MaxCurrent','Energy_Uptake','AverageVoltageDifference','AverageCurrentDifference']

    #Make new dataframe merging immediate and intermediate dataframes on ID
    df_immediateintermediate = pd.merge(df_immediate, df_intermediate, on='ID')

    #Make new dataframe merging immediateintermediate and cluster dataframes on ID
    df_immediateintermediate_clusters = pd.merge(df_immediateintermediate, df_clusters, on='ID')

    # Make a dataframe with only Half_Minutes, Charging_Half_Minutes, and ID from the final dataframe
    df_pred = df_final[['ID', 'Half_Minutes', 'Charging_Half_Minutes']].copy()

    # Merge with the 'cluster' column from df_clusters
    df_barebones = df_pred.merge(df_clusters[['ID', 'Cluster']], on='ID', how='left')

    print("Normalizing features")
    # Create MinMaxScaler instance
    scaler = MinMaxScaler()
    # Normalize features for the merged dataframes
    df_immediateintermediate[features_to_normalize] = scaler.fit_transform(df_immediateintermediate[features_to_normalize])
    df_immediateintermediate_clusters[features_to_normalize] = scaler.fit_transform(df_immediateintermediate_clusters[features_to_normalize])

    # Define the features (X) and the target variable (y) for each dataframe
    X_immediate = df_immediate.drop(['ID', 'TimeConnected'], axis=1)
    X_intermediate = df_immediateintermediate.drop(['ID', 'TimeConnected'], axis=1)
    X_clusters = df_immediateintermediate_clusters.drop(['ID', 'TimeConnected'], axis=1)
    X_barebones = df_barebones.drop(['ID', 'Half_Minutes', 'Charging_Half_Minutes'], axis=1)

    #print sizes of dataframes
    print("Size of immediate dataframe: ", X_immediate.shape)
    print("Size of intermediate dataframe: ", X_intermediate.shape)
    print("Size of immediateintermediate dataframe: ", df_immediateintermediate.shape)
    print("Size of immediateintermediate_clusters dataframe: ", X_clusters.shape)
    print("Size of clusters dataframe: ", X_clusters.shape)
    print("Size of barebones dataframe: ", X_barebones.shape)

    #Feature to predict
    y = df_pred['Half_Minutes']

    #Loop over the three prediction dataframes
    for df in [df_immediate, df_immediateintermediate, df_immediateintermediate_clusters, df_barebones]:
        #Print the first five rows of the dataframe
        print(df.head())

    # Split the data into training and testing sets
    X_immediate_train, X_immediate_test, y_train, y_test = train_test_split(X_immediate, y, test_size=test_size, random_state=random_state)
    X_intermediate_train, X_intermediate_test, _, _ = train_test_split(X_intermediate, y, test_size=test_size, random_state=random_state)
    X_clusters_train, X_clusters_test, _, _ = train_test_split(X_clusters, y, test_size=test_size, random_state=random_state)
    X_barebones_train, X_barebones_test, _, _ = train_test_split(X_barebones, y, test_size=test_size, random_state=random_state)

    # Train a machine learning model (Random Forest Regressor in this example)
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=num_cores)

    # Train the model on each dataframe
    print("Training the immediate model")
    model.fit(X_immediate_train, y_train)
    y_pred_immediate = model.predict(X_immediate_test)

    print("Training the intermediate model")
    model.fit(X_intermediate_train, y_train)
    y_pred_intermediate = model.predict(X_intermediate_test)

    print("Training the clusters model")
    model.fit(X_clusters_train, y_train)
    y_pred_clusters = model.predict(X_clusters_test)

    print("Training the barebones model")
    model.fit(X_barebones_train, y_train)
    y_pred_barebones = model.predict(X_barebones_test)

    # Evaluate the model
    print("Evaluating the models")
    mse_immediate = mean_squared_error(y_test, y_pred_immediate)
    mse_intermediate = mean_squared_error(y_test, y_pred_intermediate)
    mse_clusters = mean_squared_error(y_test, y_pred_clusters)
    mse_barebones = mean_squared_error(y_test, y_pred_barebones)

    # Print the mean squared error for each dataframe
    print(f'MSE for Immediate: {mse_immediate}')
    print(f'MSE for Intermediate: {mse_intermediate}')
    print(f'MSE for Clusters: {mse_clusters}')
    print(f'MSE for Barebones: {mse_barebones}')

    # Create a DataFrame to store the predicted values and actual values
    df_results = pd.DataFrame({'Actual': y_test, 'Immediate': y_pred_immediate, 'Intermediate': y_pred_intermediate,
                                'Clusters': y_pred_clusters, 'Barebones': y_pred_barebones})

    # Save the DataFrame to a CSV file
    results_file_name = 'predictions_vs_actuals.csv'
    df_results.to_csv(results_file_name, index=False)

    return mse_clusters
