import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def Cov():
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

    # Exclude fields
    numeric_columns = data.drop(columns=['ID', "Charging_Half_Minutes"]).select_dtypes(include=['float64', 'int64'])

    # Create covariance matrix
    covariance_matrix = numeric_columns.cov()

    # Remove diagonal elements
    np.fill_diagonal(covariance_matrix.values, np.nan)

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 12))

    # Create a heatmap with a diverging colormap
    sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, fmt=".2f")

    # Display the plot
    plt.title('Covariance Matrix Heatmap')

    # Save the figure with the current date and time in the filename
    results_dir = "plots/cov"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
    plt.close()
