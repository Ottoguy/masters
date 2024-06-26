import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import mplcursors  # Import the mplcursors library
import matplotlib.lines as mlines  # Import mlines for creating Line2D objects
from datetime import datetime

def FeaturesVHalfMinutes():
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

    # Extract only the hh:mm part from TimeDisconnected and TimeConnected
    data['TimeDisconnected'] = pd.to_datetime(data['TimeDisconnected']).dt.strftime('%H:%M')
    data['TimeConnected'] = pd.to_datetime(data['TimeConnected']).dt.strftime('%H:%M')

    # Convert TimeDisconnected and TimeConnected to datetime objects
    data['TimeDisconnected'] = pd.to_datetime(data['TimeDisconnected'], format='%H:%M')
    data['TimeConnected'] = pd.to_datetime(data['TimeConnected'], format='%H:%M')

    # Sort DataFrame chronologically based on TimeDisconnected and TimeConnected
    data.sort_values(['TimeDisconnected', 'TimeConnected'], inplace=True)

    print("CSV file loaded")

    # Columns to exclude
    exclude_columns = ['ID', 'Half_Minutes', 'Current_Type', 'FullyCharged', 'Weekend', 'ChargingPoint']

    # Columns to plot against Half_Minutes
    x_columns = [col for col in data.columns if col not in exclude_columns]

    # Calculate the number of subplots needed
    num_subplots = len(x_columns)
    num_rows = num_subplots // 2 + num_subplots % 2  # Use 2 columns for each row

    # Create subplots
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

    # Flatten the 2D array of subplots
    axes = axes.flatten()

    # Differentiate 'FullyCharged' and 'Current_Type' with colors
    colors = data.apply(lambda row: 'green' if (row['FullyCharged'] and row['Current_Type'] == '3-Phase') else
                                        'blue' if (row['FullyCharged'] and row['Current_Type'] == '1-Phase') else
                                        'red' if (not row['FullyCharged'] and row['Current_Type'] == '3-Phase') else
                                        'orange' if (not row['FullyCharged'] and row['Current_Type'] == '1-Phase') else 'black', axis=1)

    # Create legend handles and labels for each color category
    legend_handles = [
        mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='FullyCharged and 3-Phase (Weekday)'),
        mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='FullyCharged and 1-Phase (Weekday)'),
        mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Not FullyCharged and 3-Phase (Weekday)'),
        mlines.Line2D([], [], color='orange', marker='o', linestyle='None', markersize=10, label='Not FullyCharged and 1-Phase (Weekday)'),
        mlines.Line2D([], [], color='green', marker='s', linestyle='None', markersize=10, label='FullyCharged and 3-Phase (Weekend)'),
        mlines.Line2D([], [], color='blue', marker='s', linestyle='None', markersize=10, label='FullyCharged and 1-Phase (Weekend)'),
        mlines.Line2D([], [], color='red', marker='s', linestyle='None', markersize=10, label='Not FullyCharged and 3-Phase (Weekend)'),
        mlines.Line2D([], [], color='orange', marker='s', linestyle='None', markersize=10, label='Not FullyCharged and 1-Phase (Weekend)'),
    ]

    # Plotting each column against Half_Minutes with a logarithmic y-axis
    for i, x_column in enumerate(x_columns):
        row = i // 2
        col = i % 2
        
        # Separate data for weekend and non-weekend points
        weekend_data = data[data['Weekend']]
        non_weekend_data = data[~data['Weekend']]
        
        # Plot non-weekend points with circles
        scatter1 = axes[i].scatter(non_weekend_data[x_column], non_weekend_data['Half_Minutes'], c=colors[~data['Weekend']], marker='o', label='Non-Weekend')

        # Plot weekend points with squares
        scatter2 = axes[i].scatter(weekend_data[x_column], weekend_data['Half_Minutes'], c=colors[data['Weekend']], marker='s', label='Weekend')

        axes[i].set_title(f'Scatter Plot: Half_Minutes vs {x_column}')
        axes[i].set_xlabel(x_column)
        axes[i].set_ylabel('Half_Minutes')
        
        # Check if the current column is in the list of columns where you want a logarithmic x-axis
        if x_column in ['MaxVoltage', 'MaxCurrent', 'AverageVoltageDifference', "AverageCurrentDifference"]:  # Replace with the actual column names
            axes[i].set_xscale('log')  # Set x-axis to logarithmic scale
        
        axes[i].set_yscale('log')  # Set y-axis to logarithmic scale

        # Enable hover over points to display ID
        mplcursors.cursor(scatter1, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"ID: {non_weekend_data['ID'].iloc[sel.target.index]}"))
        mplcursors.cursor(scatter2, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"ID: {weekend_data['ID'].iloc[sel.target.index]}"))

        # Improve x-axis readability for TimeConnected and TimeDisconnected
        if x_column in ['TimeConnected', 'TimeDisconnected']:
            axes[i].tick_params(axis='x', rotation=0)  # No rotation
            axes[i].xaxis.set_major_locator(plt.MaxNLocator(12))  # Adjust the number of x-axis ticks
            axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))  # Format x-axis labels

    axes[-1].legend(handles=legend_handles)  # Use the last subplot for the legend, providing handles and labels

    # Adjust layout for better spacing
    plt.tight_layout()

    results_dir = "plots/features_v_half_minutes/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the figure with the current date and time in the filename
    print(f"Saving figure {current_datetime}", end='\r')
    plt.savefig(os.path.join(results_dir, current_datetime + '.png'))

    plt.close()
