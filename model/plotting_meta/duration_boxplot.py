import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def DurationBoxplot():
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

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
    meta_df = pd.read_csv(latest_file)


    #Convert half minutes to minutes
    meta_df['Half_Minutes'] = meta_df['Half_Minutes'] / 2
    meta_df['Charging_Half_Minutes'] = meta_df['Charging_Half_Minutes'] / 2

    #Rename to "Parking duration" and "Charging duration"
    meta_df = meta_df.rename(columns={'Half_Minutes': 'Parking duration', 'Charging_Half_Minutes': 'Charging duration'})

    #Define the features to plot
    features_to_plot = ['Parking duration', 'Charging duration']

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))  # 2 rows, 1 column

    # Plot each feature in a separate subplot
    for i, feature in enumerate(features_to_plot):
        axs[i].boxplot(meta_df[feature], vert=False, showmeans=False, meanline=False)

        # Add title and labels
        axs[i].set_title(f'Boxplot of {feature}', fontsize=16)
        axs[i].set_xlabel('Minutes', fontsize=14)

        # Show standard deviation on the plot
        mean = meta_df[feature].mean()
        std = meta_df[feature].std()
        #axs[i].text(mean + 0.5, 1, f'Std: {std:.2f}', ha='center', va='bottom', color='red', fontsize=14)

        # Set grid
        axs[i].grid(True)

    # Adjust layout
    plt.tight_layout()
    #increase space between subplots
    plt.subplots_adjust(hspace=0.5)

    # Show the plot
    plt.show()

# Call the function to generate separate boxplots for each feature
DurationBoxplot()
