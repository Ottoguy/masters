import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

def CovTime():
    # Specify the directory where your files are located
    folder_path = 'prints/meta/'

    # Create a pattern to match files in the specified format
    file_pattern = '*'

    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_path, file_pattern))

    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)

    # Take the latest file
    latest_file = file_list[0]

    # Load your data from the latest file
    data = pd.read_csv(latest_file)

    # Exclude the 'ID' field
    numeric_columns = data.drop(columns=['ID'])

    # Specify the time features
    time_features = ['TimeConnected_sin', 'TimeConnected_cos', 'TimeDisconnected_sin', 'TimeDisconnected_cos']

    # Separate features for the two covariance matrices
    time_and_half_minutes = numeric_columns[['Half_Minutes'] + time_features]
    time_and_charging_half_minutes = numeric_columns[['Charging_Half_Minutes'] + time_features]

    # Create covariance matrices
    cov_matrix_time_and_half_minutes = time_and_half_minutes.cov()
    cov_matrix_time_and_charging_half_minutes = time_and_charging_half_minutes.cov()

    # Set up the matplotlib figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Exclude the diagonals from the covariance matrices
    cov_matrix_time_and_half_minutes_no_diag = cov_matrix_time_and_half_minutes.copy()
    cov_matrix_time_and_half_minutes_no_diag.values[[np.arange(cov_matrix_time_and_half_minutes_no_diag.shape[0])]*2] = np.nan

    cov_matrix_time_and_charging_half_minutes_no_diag = cov_matrix_time_and_charging_half_minutes.copy()
    cov_matrix_time_and_charging_half_minutes_no_diag.values[[np.arange(cov_matrix_time_and_charging_half_minutes_no_diag.shape[0])]*2] = np.nan

    sns.heatmap(cov_matrix_time_and_half_minutes, cmap='coolwarm', annot=True, fmt=".2f", ax=axes[0], mask=np.eye(len(cov_matrix_time_and_half_minutes)))
    axes[0].set_title('Covariance Matrix (Time Features and Half_Minutes)')

    sns.heatmap(cov_matrix_time_and_charging_half_minutes, cmap='coolwarm', annot=True, fmt=".2f", ax=axes[1], mask=np.eye(len(cov_matrix_time_and_charging_half_minutes)))
    axes[1].set_title('Covariance Matrix (Time Features and Charging_Half_Minutes)')


    # Adjust layout
    plt.tight_layout()

    # Save the figure with the current date and time in the filename
    results_dir = "plots/time_cov"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, current_datetime + '_cov_matrices.png'))
    plt.close()

CovTime()