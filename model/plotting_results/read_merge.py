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

    folder2_path = 'prints/backup_oldruns/dl_merge/'
    file_pattern2 = '*'
    file_list2 = glob.glob(os.path.join(folder2_path, file_pattern2))
    file_list2.sort(key=os.path.getmtime, reverse=True)
    latest_file2 = file_list2[0]
    data2 = pd.read_csv(latest_file2)

    #Merge the two dataframes
    data = pd.concat([data, data2])

    #Delete all duplicate rows
    data = data.drop_duplicates()

    #Keep only rows with TS_Samples = 120
    #data = data[data.TS_Samples == 120]

    # Sort by lowest
    data = data.sort_values(by=['RMSE_Intermediate'])

    #pint first 5 rows
    print(data.head())
readmerge()
