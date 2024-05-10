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

    # Extract the columns you want for the boxplot
    features_to_plot = ['Half_Minutes', 'Charging_Half_Minutes']

    # Create the boxplot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    meta_df[features_to_plot].boxplot(whis='range', showmeans=True, meanline=True)

    # Add title and labels
    plt.title('Boxplot of Half_Minutes and Charging_Half_Minutes', fontsize=16)
    plt.ylabel('Minutes', fontsize=14)

    # Show standard deviation on the plot
    for i, col in enumerate(features_to_plot):
        mean = meta_df[col].mean()
        std = meta_df[col].std()
        plt.text(i + 1, mean + 0.5, f'Std: {std:.2f}', ha='center', va='bottom', color='red')

    # Show the plot
    plt.xticks(ticks=np.arange(1, len(features_to_plot) + 1), labels=features_to_plot)
    plt.grid(True)

    #Save to folder
    output_folder = 'plots/boxplot/'
    #Save as PNG with name as current_datetime
    plt.savefig(os.path.join(output_folder, f'{current_datetime}.png'))


# Call the function to generate the boxplot
DurationBoxplot()