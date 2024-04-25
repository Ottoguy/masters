import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def readmerge():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Specify the directory where your files are located
    folder_path = 'prints/dl_merge/'

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

    # Drop specified columns
    data = data.drop(columns=['RMSE_Intermediate', 'RMSE_Clusters', 'RMSE_Immediate', 'RMSE_Barebones', 'MAE_Intermediate', 'MAE_Clusters', 'MAE_Immediate', 'MAE_Barebones', 'Timestamp'])

    # Print unique values of each column
    for column in data.columns:
        print(f"Unique values in column '{column}':")
        unique_values = data[column].unique()
        for value in unique_values:
            print(value)
        print("\n")

readmerge()
