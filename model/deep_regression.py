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
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Flatten
from datetime import datetime
from sklearn.metrics import mean_absolute_error

def DeepLearningRegression(ts_samples, clusters, test_size, random_state,
                            epochs, batch_size, layer1_units, layer2_units, layer3_units, dropout_rate, feature_to_exclude,
                            layer1activation, layer2activation, layer3activation, should_embed,
                            train_immediate, train_barebones):
    print("Loading data for regression")

    settings = "samples_" + str(ts_samples) + "_clusters_" + str(clusters) + "_test_size_" + str(test_size) + "_epochs_" + str(epochs) + "_batch_" + str(batch_size) + "_l1_u_" + str(layer1_units) + "_l2_u_" + str(layer2_units) + "_l3_u_" + str(layer3_units) + "_dropout_" + str(dropout_rate) + "_exclude_" + feature_to_exclude + "_l1_a_" + layer1activation + "_l2_a_" + layer2activation + "_l3_a_" + layer3activation + "_embed_" + str(should_embed)

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

    origin_1_phase = "1-Phase_" + str(clusters) + "_clusters"
    origin_3_phase = "3-Phase_" + str(clusters) + "_clusters"
    ts_cluster_1_phase_folder = 'prints/ts_clustering/' + str(ts_samples) + "/" + origin_1_phase + '/'
    ts_cluster_3_phase_folder = 'prints/ts_clustering/' + str(ts_samples) + "/" + origin_3_phase + '/'

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

    #Make a new df called "df_cluster_meta" which contains the first row of df_cluster
    df_cluster_meta = df_clusters.head(1)
    #Make a string out of this row
    cluster_meta = df_cluster_meta.to_string(index=False, header=False)

    #Remove all rows from df_clusters that are not "ID" or "Clusters"
    df_clusters = df_clusters[['ID', 'Cluster']]
    
    #Print how many rows were removed
    print("Removed", len(df_immediate) - len(df_clusters), "rows from the immediate dataframe (filtered in preprocessing or because of phase?)")

    # List of features to normalize
    features_to_normalize = ['MaxVoltage','MaxCurrent','Energy_Uptake','AverageVoltageDifference','AverageCurrentDifference', 'TimeConnected_sin', 'TimeConnected_cos']
    immediate_features_to_normalize = ['TimeConnected_sin', 'TimeConnected_cos']

    # One-hot encode the 'Weekend' column if it exists
    if 'Weekend' in df_immediate.columns:
        df_immediate = pd.get_dummies(df_immediate, columns=['Weekend'])

    if 'Weekend' in df_intermediate.columns:
        df_intermediate = pd.get_dummies(df_intermediate, columns=['Weekend'])

    if 'Weekend' in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=['Weekend'])

    # One-hot encode the 'FullyCharged' column if it exists
    if 'FullyCharged' in df_immediate.columns:
        df_immediate = pd.get_dummies(df_immediate, columns=['FullyCharged'])

    if 'FullyCharged' in df_intermediate.columns:
        df_intermediate = pd.get_dummies(df_intermediate, columns=['FullyCharged'])

    if 'FullyCharged' in df_final.columns:
        df_final = pd.get_dummies(df_final, columns=['FullyCharged'])

    #Make new dataframe merging immediate and intermediate dataframes on ID
    df_immediateintermediate = pd.merge(df_immediate, df_intermediate, on='ID')

    #Make new dataframe merging immediateintermediate and cluster dataframes on ID
    df_immediateintermediate_clusters = pd.merge(df_immediateintermediate, df_clusters, on='ID')

    # Make a dataframe with only Half_Minutes, Charging_Half_Minutes, and ID from the final dataframe
    df_pred = df_final[['ID', 'Half_Minutes', 'Charging_Half_Minutes']].copy()

    # Merge with the 'cluster' column from df_clusters
    if train_barebones:
        df_barebones = df_pred.merge(df_clusters[['ID', 'Cluster']], on='ID', how='left')

    print("Normalizing features")
    # Create MinMaxScaler instance
    scaler = MinMaxScaler()
    # Normalize features for the merged dataframes
    if train_immediate:
        df_immediate[immediate_features_to_normalize] = scaler.fit_transform(df_immediate[immediate_features_to_normalize])
    df_immediateintermediate[features_to_normalize] = scaler.fit_transform(df_immediateintermediate[features_to_normalize])
    df_immediateintermediate_clusters[features_to_normalize] = scaler.fit_transform(df_immediateintermediate_clusters[features_to_normalize])

    #Sort all dataframes by ID
    if train_immediate:
        df_immediate = df_immediate.sort_values(by='ID')
    df_intermediate = df_intermediate.sort_values(by='ID')
    df_final = df_final.sort_values(by='ID')
    df_clusters = df_clusters.sort_values(by='ID')
    df_immediateintermediate = df_immediateintermediate.sort_values(by='ID')
    df_immediateintermediate_clusters = df_immediateintermediate_clusters.sort_values(by='ID')
    df_pred = df_pred.sort_values(by='ID')
    if train_barebones:
        df_barebones = df_barebones.sort_values(by='ID')

    # Exclude one feature at a time if it exists in the dataframe
    print("Feature to exclude:", feature_to_exclude)
    if train_immediate:
        if feature_to_exclude in df_immediate.columns:
            df_immediate_excluded = df_immediate.drop([feature_to_exclude], axis=1)
        else:
            df_immediate_excluded = df_immediate

    if feature_to_exclude in df_immediateintermediate.columns:
        df_immediateintermediate_excluded = df_immediateintermediate.drop([feature_to_exclude], axis=1)
    else:
        df_immediateintermediate_excluded = df_immediateintermediate

    if feature_to_exclude in df_immediateintermediate_clusters.columns:
        df_immediateintermediate_clusters_excluded = df_immediateintermediate_clusters.drop([feature_to_exclude], axis=1)
    else:
        df_immediateintermediate_clusters_excluded = df_immediateintermediate_clusters

    if train_barebones:
        if feature_to_exclude in df_barebones.columns:
            df_barebones_excluded = df_barebones.drop([feature_to_exclude], axis=1)
        else:
            df_barebones_excluded = df_barebones

    # Define the features (X) and the target variable (y) for each dataframe
    if train_immediate:
        X_immediate_excluded = df_immediate_excluded.drop(['ID', 'TimeConnected'], axis=1)
    X_intermediate_excluded = df_immediateintermediate_excluded.drop(['ID', 'TimeConnected'], axis=1)
    X_clusters_excluded = df_immediateintermediate_clusters_excluded.drop(['ID', 'TimeConnected'], axis=1)
    if train_barebones:
        X_barebones_excluded = df_barebones_excluded.drop(['ID', 'Half_Minutes', 'Charging_Half_Minutes'], axis=1)

    #Feature to predict
    y = df_pred['Charging_Half_Minutes']

    if train_immediate:
        X_immediate_excluded = np.asarray(X_immediate_excluded).astype('float32')
        input_dim_immediate = X_immediate_excluded.shape[1]
    X_intermediate_excluded = np.asarray(X_intermediate_excluded).astype('float32')
    input_dim_intermediate = X_intermediate_excluded.shape[1]
    X_clusters_excluded = np.asarray(X_clusters_excluded).astype('float32')
    input_dim_clusters = X_clusters_excluded.shape[1]
    if train_barebones:
        X_barebones_excluded = np.asarray(X_barebones_excluded).astype('float32')
        input_dim_barebones = X_barebones_excluded.shape[1]
    y = np.asarray(y).astype('float32')
    
    def build_model(input_dim):
        model = Sequential()
        model.add(Dense(layer1_units, input_dim=input_dim, activation=layer1activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer2_units, activation=layer2activation))
        model.add(Dense(layer3_units, activation=layer3activation))
        model.add(Dense(1, activation='linear'))  # Output layer for regression
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def build_model_with_embedding(input_dim, num_categories, embedding_dim):
        model = Sequential()
        model.add(Embedding(input_dim=num_categories, output_dim=embedding_dim))
        model.add(Flatten())
        model.add(Dense(layer1_units, input_dim=input_dim, activation=layer1activation))
        model.add(Dropout(dropout_rate))
        model.add(Dense(layer2_units, activation=layer2activation))
        model.add(Dense(layer3_units, activation=layer3activation))
        model.add(Dense(1, activation='linear'))  # Output layer for regression
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    # Define the number of categories and embedding dimension
    num_categories = len(df_immediateintermediate_clusters['Cluster'].unique())
    #Normal rule of thumb
    embedding_dim = min(50, num_categories // 2)

    # Build models
    if train_immediate:
        model_immediate = build_model(input_dim_immediate)
    model_intermediate = build_model(input_dim_intermediate)

    if should_embed:
        model_clusters = build_model_with_embedding(input_dim_clusters, num_categories, embedding_dim)
        if train_barebones:
            model_barebones = build_model_with_embedding(input_dim_barebones, num_categories, embedding_dim)
    else:
        model_clusters = build_model(input_dim_clusters)
        if train_barebones:
            model_barebones = build_model(input_dim_barebones)

    # Split the data for each set separately
    if train_immediate:
        X_immediate_train_excluded, X_immediate_test_excluded, y_immediate_train, y_immediate_test = train_test_split(X_immediate_excluded, y, test_size=test_size, random_state=random_state)
    X_intermediate_train_excluded, X_intermediate_test_excluded, y_intermediate_train, y_intermediate_test = train_test_split(X_intermediate_excluded, y, test_size=test_size, random_state=random_state)
    X_clusters_train_excluded, X_clusters_test_excluded, y_clusters_train, y_clusters_test = train_test_split(X_clusters_excluded, y, test_size=test_size, random_state=random_state)
    if train_barebones:
        X_barebones_train_excluded, X_barebones_test_excluded, y_barebones_train, y_barebones_test = train_test_split(X_barebones_excluded, y, test_size=test_size, random_state=random_state)

    # Train the models
    if train_immediate:
        print(f"Training the immediate model with {feature_to_exclude} excluded")
        model_immediate.fit(X_immediate_train_excluded, y_immediate_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_immediate = model_immediate.predict(X_immediate_test_excluded).flatten()

    print(f"Training the intermediate model with {feature_to_exclude} excluded")
    model_intermediate.fit(X_intermediate_train_excluded, y_intermediate_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_intermediate = model_intermediate.predict(X_intermediate_test_excluded).flatten()

    print(f"Training the clusters model with {feature_to_exclude} excluded")
    model_clusters.fit(X_clusters_train_excluded, y_clusters_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred_clusters = model_clusters.predict(X_clusters_test_excluded).flatten()

    if train_barebones:
        print(f"Training the barebones model with {feature_to_exclude} excluded")
        model_barebones.fit(X_barebones_train_excluded, y_barebones_train, epochs=epochs, batch_size=batch_size, verbose=0)
        y_pred_barebones = model_barebones.predict(X_barebones_test_excluded).flatten()

    # Evaluate the models
    print("Evaluating the models")
    if train_immediate:
        mse_immediate = mean_squared_error(y_immediate_test, y_pred_immediate)
        mae_immediate = mean_absolute_error(y_immediate_test, y_pred_immediate)
        rmse_immediate = np.sqrt(mse_immediate)
    else:
        mse_immediate = None
        mae_immediate = None
        rmse_immediate = None

    mse_intermediate = mean_squared_error(y_intermediate_test, y_pred_intermediate)
    mae_intermediate = mean_absolute_error(y_intermediate_test, y_pred_intermediate)
    rmse_intermediate = np.sqrt(mse_intermediate)

    mse_clusters = mean_squared_error(y_clusters_test, y_pred_clusters)
    mae_clusters = mean_absolute_error(y_clusters_test, y_pred_clusters)
    rmse_clusters = np.sqrt(mse_clusters)

    if train_barebones:
        mse_barebones = mean_squared_error(y_barebones_test, y_pred_barebones)
        mae_barebones = mean_absolute_error(y_barebones_test, y_pred_barebones)
        rmse_barebones = np.sqrt(mse_barebones)
    else:
        mse_barebones = None
        mae_barebones = None
        rmse_barebones = None

    # Write the real values and the predicted values by all models to a csv file
    if train_immediate:
        if train_barebones:
            df_results_all = pd.DataFrame({'Real': y_immediate_test, 'Immediate': y_pred_immediate, 'Intermediate': y_pred_intermediate, 'Clusters': y_pred_clusters, 'Barebones': y_pred_barebones})
        else:
            df_results_all = pd.DataFrame({'Real': y_immediate_test, 'Immediate': y_pred_immediate, 'Intermediate': y_pred_intermediate, 'Clusters': y_pred_clusters})
    else:
        if train_barebones:
            df_results_all = pd.DataFrame({'Real': y_intermediate_test, 'Intermediate': y_pred_intermediate, 'Clusters': y_pred_clusters, 'Barebones': y_pred_barebones})
        else:
            df_results_all = pd.DataFrame({'Real': y_intermediate_test, 'Intermediate': y_pred_intermediate, 'Clusters': y_pred_clusters})
            
    #Sort df_result by value of 'Real'
    df_results_all = df_results_all.sort_values(by='Real')

    output_folder = 'prints/deep_learning'
    # Create a folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the file name
    print(f"Creating the file {current_datetime}_{settings}.csv")
    output_file = f"{output_folder}/{current_datetime}_{settings}.csv"
    # Print desired_rows to a CSV file
    df_results_all.to_csv(output_file, index=False)
    #Print path to the created file
    print(f"Results saved to {output_file}")

    return rmse_barebones, rmse_immediate, rmse_intermediate, rmse_clusters, mae_barebones, mae_immediate, mae_intermediate, mae_clusters, cluster_meta