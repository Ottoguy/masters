import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

def DeepRegression(num_cores, ts_samples, include_ts_clusters, phase, clusters, test_size, random_state, epochs=50, batch_size=32):
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

        #save name of cluster file to string
        cluster_file_name = latest_id_cluster_file.split("/")[-1]
        cluster_file_name = cluster_file_name.split(".")[0]

    #Make new dataframe merging immediate and intermediate dataframes on ID
    df_immediateintermediate = pd.merge(df_immediate, df_intermediate, on='ID')

    #Make new dataframe merging immediateintermediate and cluster dataframes on ID
    df_immediateintermediate_clusters = pd.merge(df_immediateintermediate, df_clusters, on='ID')

    #Make a dataframe with only Half_Minutes, Half_Charging_Minutes, and ID from the final dataframe and merge with cluster from df_clusters
    df_pred = df_final[['ID', 'Half_Minutes', 'Half_Charging_Minutes']].copy
    df_barebones = pd.merge(df_pred, df_clusters, on='ID')
    #Drop the Silhouette Score and Num Clusters columns
    df_barebones = df_barebones.drop(['Silhouette Score', 'Num Clusters'], axis=1)

    # Define the features (X) and the target variable (y) for each dataframe
    X_immediate = df_immediate.drop(['ID', 'TimeConnected'], axis=1)
    X_intermediate = df_immediateintermediate.drop(['ID', 'TimeConnected'], axis=1)
    X_clusters = df_immediateintermediate_clusters.drop(['ID', 'TimeConnected'], axis=1)
    X_barebones = df_barebones.drop(['ID', 'Charging_Half_Minutes'], axis=1)

    #Feature to predict
    y = df_final['Half_Minutes']

    # Normalize the data using StandardScaler
    scaler = StandardScaler()

    X_immediate_train_scaled = scaler.fit_transform(X_immediate_train)
    X_intermediate_train_scaled = scaler.fit_transform(X_intermediate_train)
    X_clusters_train_scaled = scaler.fit_transform(X_clusters_train)
    X_barebones_train_scaled = scaler.fit_transform(X_barebones_train)

    # Build a neural network model
    def build_model(input_shape):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))  # Output layer with one neuron for regression
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    # Get the number of features for each dataframe
    input_shape_immediate = X_immediate_train_scaled.shape[1]
    input_shape_intermediate = X_intermediate_train_scaled.shape[1]
    input_shape_clusters = X_clusters_train_scaled.shape[1]
    input_shape_barebones = X_barebones_train_scaled.shape[1]

    # Build models
    model_immediate = build_model(input_shape_immediate)
    model_intermediate = build_model(input_shape_intermediate)
    model_clusters = build_model(input_shape_clusters)
    model_barebones = build_model(input_shape_barebones)

    # Train models
    model_immediate.fit(X_immediate_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    model_intermediate.fit(X_intermediate_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    model_clusters.fit(X_clusters_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    model_barebones.fit(X_barebones_train_scaled, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predictions
    y_pred_immediate = model_immediate.predict(X_immediate_test)
    y_pred_intermediate = model_intermediate.predict(X_intermediate_test)
    y_pred_clusters = model_clusters.predict(X_clusters_test)
    y_pred_barebones = model_barebones.predict(X_barebones_test)

    # Inverse transform predictions to original scale
    y_pred_immediate = scaler.inverse_transform(y_pred_immediate)
    y_pred_intermediate = scaler.inverse_transform(y_pred_intermediate)
    y_pred_clusters = scaler.inverse_transform(y_pred_clusters)
    y_pred_barebones = scaler.inverse_transform(y_pred_barebones)

    # Evaluate the model
    mse_immediate = mean_squared_error(y_test, y_pred_immediate)
    mse_intermediate = mean_squared_error(y_test, y_pred_intermediate)
    mse_clusters = mean_squared_error(y_test, y_pred_clusters)
    mse_barebones = mean_squared_error(y_test, y_pred_barebones)

    # Print the mean squared error for each dataframe
    print(f'MSE for Immediate: {mse_immediate}')
    print(f'MSE for Intermediate: {mse_intermediate}')
    print(f'MSE for Clusters: {mse_clusters}')
    print(f'MSE for Barebones: {mse_barebones}')

    # Regression plot for Immediate
    sns.regplot(x=y_test, y=y_pred_immediate.flatten(), scatter_kws={'s': 10})
    plt.title('Regression Plot for Immediate')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

    # Regression plot for Intermediate
    sns.regplot(x=y_test, y=y_pred_intermediate.flatten(), scatter_kws={'s': 10})
    plt.title('Regression Plot for Intermediate')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

    # Regression plot for Clusters
    sns.regplot(x=y_test, y=y_pred_clusters.flatten(), scatter_kws={'s': 10})
    plt.title('Regression Plot for Clusters')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

    # Regression plot for Barebones
    sns.regplot(x=y_test, y=y_pred_barebones.flatten(), scatter_kws={'s': 10})
    plt.title('Regression Plot for Barebones')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

# Example usage
num_cores = -1
ts_samples = 100
include_ts_clusters = True
phase = "example"
clusters = 3
test_size = 0.2
random_state = 42
n_estimators = 100
DeepLearningRegression(num_cores, ts_samples, include_ts_clusters, phase, clusters, test_size, random_state)