import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Dropout
from datetime import datetime

def DeepLearningRegression(num_cores, ts_samples, include_ts_clusters, clusters, test_size, random_state,
                            epochs, batch_size, layer1_units, layer2_units, dropout_rate, feature_to_exclude):
    print("Loading data for regression")

    settings = "ts_samples_" + str(ts_samples) + "_clusters_" + str(clusters) + "_test_size_" + str(test_size) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_layer1_units_" + str(layer1_units) + "_layer2_units_" + str(layer2_units) + "_dropout_rate_" + str(dropout_rate)

    # Specify the directory where your files are located
    folder_immediate_path = 'prints/preproc_immediate/'
    folder_intermediate_path = 'prints/preproc_intermediate/'
    folder_final_path = 'prints/preproc_final/'

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
        origin_1_phase = "1-Phase_" + str(clusters) + "_clusters"
        origin_3_phase = "3-Phase_" + str(clusters) + "_clusters"
        ts_cluster_1_phase_folder = 'prints/ts_clustering/' + str(ts_samples) + "/" + origin_1_phase + '/'
        ts_cluster_3_phase_folder = 'prints/ts_clustering/' + str(ts_samples) + "/" + origin_3_phase + '/'
        cluster_file_name = ""

        # Get a list of all files in the specified format within the chosen subfolder for 1-phase
        id_cluster_1_phase_files = glob.glob(os.path.join(ts_cluster_1_phase_folder, '*.csv'))
        # Sort the files based on modification time (latest first)
        id_cluster_1_phase_files.sort(key=os.path.getmtime, reverse=True)
        # Take the latest file from the chosen subfolder
        latest_id_cluster_1_phase_file = id_cluster_1_phase_files[0]
        # Load your ID-cluster mapping data from the latest file
        df_clusters_1_phase = pd.read_csv(latest_id_cluster_1_phase_file)

        # Get a list of all files in the specified format within the chosen subfolder for 3-phase
        id_cluster_3_phase_files = glob.glob(os.path.join(ts_cluster_3_phase_folder, '*.csv'))
        # Sort the files based on modification time (latest first)
        id_cluster_3_phase_files.sort(key=os.path.getmtime, reverse=True)
        # Take the latest file from the chosen subfolder
        latest_id_cluster_3_phase_file = id_cluster_3_phase_files[0]
        #load your ID-cluster mapping data from the latest file
        df_clusters_3_phase = pd.read_csv(latest_id_cluster_3_phase_file)

        #Merge the two dataframes sorted by ID
        df_clusters = pd.concat([df_clusters_1_phase, df_clusters_3_phase], ignore_index=True)
        #Replace NaN values with -1 (some ID's have not been clustered?)
        df_clusters['Cluster'] = df_clusters['Cluster'].fillna(-1)

        #Remove all rows from the other dataframes with ids that are not in the cluster dataframe
        df_immediate = df_immediate[df_immediate['ID'].isin(df_clusters['ID'])]
        df_intermediate = df_intermediate[df_intermediate['ID'].isin(df_clusters['ID'])]
        df_final = df_final[df_final['ID'].isin(df_clusters['ID'])]
        
        #Print how many rows were removed
        print("Removed", len(df_immediate) - len(df_clusters), "rows from the immediate dataframe (filtered in preprocessing or because of phase?)")

    # List of features to normalize
    features_to_normalize = ['MaxVoltage','MaxCurrent','Energy_Uptake','AverageVoltageDifference','AverageCurrentDifference', 'TimeConnected_sin', 'TimeConnected_cos']
    immediate_features_to_normalize = ['TimeConnected_sin', 'TimeConnected_cos']

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
    df_immediate[immediate_features_to_normalize] = scaler.fit_transform(df_immediate[immediate_features_to_normalize])
    df_immediateintermediate[features_to_normalize] = scaler.fit_transform(df_immediateintermediate[features_to_normalize])
    df_immediateintermediate_clusters[features_to_normalize] = scaler.fit_transform(df_immediateintermediate_clusters[features_to_normalize])

    #Sort all dataframes by ID
    df_immediate = df_immediate.sort_values(by='ID')
    df_intermediate = df_intermediate.sort_values(by='ID')
    df_final = df_final.sort_values(by='ID')
    df_clusters = df_clusters.sort_values(by='ID')
    df_immediateintermediate = df_immediateintermediate.sort_values(by='ID')
    df_immediateintermediate_clusters = df_immediateintermediate_clusters.sort_values(by='ID')
    df_pred = df_pred.sort_values(by='ID')
    df_barebones = df_barebones.sort_values(by='ID')

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
    y = df_pred['Charging_Half_Minutes']

    # Define the neural network model
    def build_model(input_dim):
        model = Sequential()
        model.add(Dense(layer1_units, input_dim=input_dim, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer2_units, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Output layer for regression
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    X_immediate = np.asarray(X_immediate).astype('float32')
    X_intermediate = np.asarray(X_intermediate).astype('float32')
    X_clusters = np.asarray(X_clusters).astype('float32')
    X_barebones = np.asarray(X_barebones).astype('float32')
    y = np.asarray(y).astype('float32')

    #print length of the numpy arrays
    print("Length of X_immediate: ", len(X_immediate))
    print("Length of X_intermediate: ", len(X_intermediate))
    print("Length of X_clusters: ", len(X_clusters))
    print("Length of X_barebones: ", len(X_barebones))
    print("Length of y: ", len(y))

    # Exclude one feature at a time if it exists in the dataframe
    print("Feature to exclude:", feature_to_exclude)
    if feature_to_exclude in df_immediate.columns:
        excluded_feature_index_immediate = df_immediate.columns.get_loc(feature_to_exclude)
        X_immediate_excluded = np.delete(X_immediate, excluded_feature_index_immediate, axis=1)
    else:
        X_immediate_excluded = X_immediate.copy()

    if feature_to_exclude in df_immediateintermediate.columns:
        excluded_feature_index_intermediate = df_immediateintermediate.columns.get_loc(feature_to_exclude)
        X_intermediate_excluded = np.delete(X_intermediate, excluded_feature_index_intermediate, axis=1)
    else:
        X_intermediate_excluded = X_intermediate.copy()

    if feature_to_exclude in df_immediateintermediate_clusters.columns:
        excluded_feature_index_clusters = df_immediateintermediate_clusters.columns.get_loc(feature_to_exclude)
        X_clusters_excluded = np.delete(X_clusters, excluded_feature_index_clusters, axis=1)
    else:
        X_clusters_excluded = X_clusters.copy()

    if feature_to_exclude in df_barebones.columns:
        excluded_feature_index_barebones = df_barebones.columns.get_loc(feature_to_exclude)
        X_barebones_excluded = np.delete(X_barebones, excluded_feature_index_barebones, axis=1)
    else:
        X_barebones_excluded = X_barebones.copy()

    # Get the input dimensions for the neural network
    input_dim_immediate = X_immediate_excluded.shape[1]
    input_dim_intermediate = X_intermediate_excluded.shape[1]
    input_dim_clusters = X_clusters_excluded.shape[1]
    input_dim_barebones = X_barebones_excluded.shape[1]

    # Build models
    model_immediate = build_model(input_dim_immediate)
    model_intermediate = build_model(input_dim_intermediate)
    model_clusters = build_model(input_dim_clusters)
    model_barebones = build_model(input_dim_barebones)

    # Split the data into training and testing sets
    X_immediate_train_excluded, X_immediate_test_excluded, y_train, y_test = train_test_split(X_immediate_excluded, y, test_size=test_size, random_state=random_state)
    X_intermediate_train_excluded, X_intermediate_test_excluded, _, _ = train_test_split(X_intermediate_excluded, y, test_size=test_size, random_state=random_state)
    X_clusters_train_excluded, X_clusters_test_excluded, _, _ = train_test_split(X_clusters_excluded, y, test_size=test_size, random_state=random_state)
    X_barebones_train_excluded, X_barebones_test_excluded, _, _ = train_test_split(X_barebones_excluded, y, test_size=test_size, random_state=random_state)

    #Print shapes of the training and testing sets
    print("Shape of X_immediate_train_excluded: ", X_immediate_train_excluded.shape)
    print("Shape of X_immediate_test_excluded: ", X_immediate_test_excluded.shape)
    print("Shape of X_intermediate_train_excluded: ", X_intermediate_train_excluded.shape)
    print("Shape of X_intermediate_test_excluded: ", X_intermediate_test_excluded.shape)
    print("Shape of X_clusters_train_excluded: ", X_clusters_train_excluded.shape)
    print("Shape of X_clusters_test_excluded: ", X_clusters_test_excluded.shape)
    print("Shape of X_barebones_train_excluded: ", X_barebones_train_excluded.shape)
    print("Shape of X_barebones_test_excluded: ", X_barebones_test_excluded.shape)
    print("Shape of y_train: ", y_train.shape)
    print("Shape of y_test: ", y_test.shape)

    # Train the models
    print(f"Training the immediate model with {feature_to_exclude} excluded")
    model_immediate.fit(X_immediate_train_excluded, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_immediate = model_immediate.predict(X_immediate_test_excluded).flatten()

    print(f"Training the intermediate model with {feature_to_exclude} excluded")
    y_pred_intermediate = model_intermediate.predict(X_intermediate_test_excluded).flatten()

    print(f"Training the clusters model with {feature_to_exclude} excluded")
    model_clusters.fit(X_clusters_train_excluded, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_clusters = model_clusters.predict(X_clusters_test_excluded).flatten()

    print(f"Training the barebones model with {feature_to_exclude} excluded")
    model_barebones.fit(X_barebones_train_excluded, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_barebones = model_barebones.predict(X_barebones_test_excluded).flatten()

    # Evaluate the models
    print("Evaluating the models")
    mse_immediate = mean_squared_error(y_test, y_pred_immediate)
    mse_intermediate = mean_squared_error(y_test, y_pred_intermediate)
    mse_clusters = mean_squared_error(y_test, y_pred_clusters)
    mse_barebones = mean_squared_error(y_test, y_pred_barebones)

    # Write the real values and the predicted values by all models to a csv file
    df_results = pd.DataFrame({'Real': y_test, 'Immediate': y_pred_immediate,
                                'Intermediate': y_pred_intermediate, 'Clusters': y_pred_clusters,
                                'Barebones': y_pred_barebones, 'ExcludedFeature': feature_to_exclude})
        
    df_results_all = df_results

    #Sort df_result by value of 'Real'
    df_results_all = df_results_all.sort_values(by='Real')

    print(len(df_results_all))

    output_folder = 'prints/deep_learning'
    # Create a folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the file name
    print(f"Creating the file {settings}_{current_datetime}.csv")
    output_file = f"{output_folder}/{settings}_{current_datetime}.csv"
    # Print desired_rows to a CSV file
    df_results_all.to_csv(output_file, index=False)
    #Print path to the created file
    print(f"Results saved to {output_file}")

    return mse_barebones, mse_immediate, mse_intermediate, mse_clusters