import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def ConnectionDurationA(threshold):
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

    # Filter out EVs with values over the threshold
    filtered_meta_df = meta_df[meta_df['Half_Minutes'] <= threshold]
    
    # Extract the 'Half_Minutes' and 'Type' columns from the filtered DataFrame
    connection_durations_1phase_all = filtered_meta_df[filtered_meta_df['Current_Type'] == '1-Phase']['Half_Minutes']
    connection_durations_3phase_all = filtered_meta_df[filtered_meta_df['Current_Type'] == '3-Phase']['Half_Minutes']

    # Filter out EVs with FullyCharged = False
    filtered_3phase_not_fully_charged = filtered_meta_df[(filtered_meta_df['Current_Type'] == '3-Phase') & (filtered_meta_df['FullyCharged'] == False)]
    filtered_1phase_not_fully_charged = filtered_meta_df[(filtered_meta_df['Current_Type'] == '1-Phase') & (filtered_meta_df['FullyCharged'] == False)]

    # Filter out EVs with FullyCharged = True
    filtered_3phase_fully_charged = filtered_meta_df[(filtered_meta_df['Current_Type'] == '3-Phase') & (filtered_meta_df['FullyCharged'] == True)]
    filtered_1phase_fully_charged = filtered_meta_df[(filtered_meta_df['Current_Type'] == '1-Phase') & (filtered_meta_df['FullyCharged'] == True)]

    disregarded_count = len(meta_df) - len(filtered_meta_df)

    # Convert durations to hours and minutes
    connection_durations_1phase_hours_minutes_all = connection_durations_1phase_all / 2 / 60
    connection_durations_3phase_hours_minutes_all = connection_durations_3phase_all / 2 / 60
    connection_durations_1phase_hours_minutes_not_fully_charged = filtered_1phase_not_fully_charged['Half_Minutes'] / 2 / 60
    connection_durations_3phase_hours_minutes_not_fully_charged = filtered_3phase_not_fully_charged['Half_Minutes'] / 2 / 60
    connection_durations_1phase_hours_minutes_fully_charged = filtered_1phase_fully_charged['Half_Minutes'] / 2 / 60
    connection_durations_3phase_hours_minutes_fully_charged = filtered_3phase_fully_charged['Half_Minutes'] / 2 / 60

    # Set the bins and range for both histograms in hours and minutes
    bins = 50
    range_vals_hours_minutes = (
        min(connection_durations_1phase_hours_minutes_all.min(), connection_durations_3phase_hours_minutes_all.min()),
        max(connection_durations_1phase_hours_minutes_all.max(), connection_durations_3phase_hours_minutes_all.max())
    )

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot the distribution with the bottom of bars as red for 3-Phase (Not Fully Charged) and blue for 3-Phase (Fully Charged)
    counts_3phase_fully_charged, bins_3phase_fully_charged, _ = axes[0].hist(connection_durations_3phase_hours_minutes_fully_charged, bins=bins, range=range_vals_hours_minutes, color='blue', edgecolor='black', alpha=1, label='3-Phase (Fully Charged)', histtype='barstacked', stacked=True)
    counts_1phase_fully_charged, bins_1phase_fully_charged, _ = axes[0].hist(connection_durations_1phase_hours_minutes_fully_charged, bins=bins, range=range_vals_hours_minutes, color='green', edgecolor='black', alpha=1, label='1-Phase (Fully Charged)', histtype='barstacked', stacked=True)

    # Set title and legend for the first subplot
    axes[0].set_title('Connection Durations for Fully Charged EVs')
    axes[0].legend()

    # Plot the distribution with the bottom of bars as orange for 1-Phase (Not Fully Charged) and green for 1-Phase (Fully Charged)
    counts_3phase_not_fully_charged, bins_3phase_not_fully_charged, _ = axes[1].hist(connection_durations_3phase_hours_minutes_not_fully_charged, bins=bins, range=range_vals_hours_minutes, color='red', edgecolor='black', alpha=1, label='3-Phase (Not Fully Charged)', histtype='barstacked', stacked=True)
    counts_1phase_not_fully_charged, bins_1phase_not_fully_charged, _ = axes[1].hist(connection_durations_1phase_hours_minutes_not_fully_charged, bins=bins, range=range_vals_hours_minutes, color='orange', edgecolor='black', alpha=1, label='1-Phase (Not Fully Charged)', histtype='barstacked', stacked=True)

    # Set title and legend for the second subplot
    axes[1].set_title('Connection Durations for Not Fully Charged EVs')
    axes[1].legend()

    # Find the maximum count among all histograms
    max_count = max([counts_3phase_not_fully_charged.max(), counts_3phase_fully_charged.max(), counts_1phase_not_fully_charged.max(), counts_1phase_fully_charged.max()])

    # Set common y-axis range for all subplots
    for ax in axes:
        ax.set_ylim(0, max_count + 5)

    # Set common labels
    for ax in axes:
        ax.set(xlabel='Connection Duration (Hours)', ylabel='Number of EVs')
        ax.grid(True, which='both', linestyle=':', linewidth=0.3, color='gray', alpha=0.5)
        ax.minorticks_on()

    # Add legend for disregarded EVs and types
    legend_text = f'Omitted EVs (>{threshold/120} hours): {disregarded_count}'
    fig.legend([legend_text], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=1, fancybox=True, shadow=True)

    plt.annotate(f"Created: {current_datetime}", xy=(10, 10), xycoords="figure pixels", fontsize=8, color='dimgray')

    results_dir = "plots/connection_duration_a/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Save the figure with the current date and time in the filename
    print(f"Saving figure {current_datetime}", end='\r')
    plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
    plt.close()