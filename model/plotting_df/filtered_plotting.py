import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from os import path
from datetime import datetime
import glob
import csv
from matplotlib.lines import Line2D

# Get the list of CSV files in the directory
csv_files = glob.glob('prints/filtered/*.csv')

results_dir = "plots/filtered_df/"

# Sort the files based on modification time and get the latest file
latest_csv_file = max(csv_files, key=os.path.getmtime)

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# Read data from the latest CSV file
data = []

with open(latest_csv_file, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        data.append(row)

data = data[1:]  # Remove the header row

def create_plot(id_value, data):
    # Create a unique plot name based on ID
    plot_name = f"{id_value}.png"

    if path.exists(results_dir + "/" + plot_name):
        print("File " + plot_name + "exists, skipping...", end='\r')
        return
    else:
        timestamps = []
        status = []  # Added for Online/Offline status'
        days = []
        day = []
        day_count = 0
        phase1_current = []
        phase2_current = []
        phase3_current = []
        phase1_voltage = []
        phase2_voltage = []
        phase3_voltage = []
        for row in data:
            # Extract relevant data for plotting
            timestamp = row[1][11:-7]  # Assuming the timestamp is in the second column
            day = row[1][8:9]
            if day not in days:
                days.append(day)
                day_count += 1
            timestamps.append(str(day_count) + "-" + str(timestamp))
            phase1_current.append(float(row[2]))
            phase2_current.append(float(row[3]))
            phase3_current.append(float(row[4]))
            phase1_voltage.append(float(row[5]))
            phase2_voltage.append(float(row[6]))
            phase3_voltage.append(float(row[7]))
            status.append(row[8])
        print("Creating figure " + plot_name, end='\r')

        # Create subplots using gridspec
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[8, 8, 1], hspace=0.4, wspace=0.0)

        # Plot the currents in the first subplot
        ax1 = plt.subplot(gs[0])
        ax1.plot(timestamps, phase1_current, label="Phase 1", linestyle='--')
        ax1.plot(timestamps, phase2_current, label="Phase 2", linestyle='-.')
        ax1.plot(timestamps, phase3_current, label="Phase 3", linestyle=':')
        ax1.set_ylabel('Current (A)')
        ax1.legend()
        ax1.margins(x=0)

        # Plot the voltages in the second subplot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(timestamps, phase1_voltage, label="Phase 1", linestyle='--')
        ax2.plot(timestamps, phase2_voltage, label="Phase 2", linestyle='-.')
        ax2.plot(timestamps, phase3_voltage, label="Phase 3", linestyle=':')
        ax2.set_ylabel('Voltage (V)')
        ax2.legend()
        ax2.margins(x=0)

        # Add unintrusive grid to both subplots
        ax1.grid(alpha=0.3, linestyle='--')
        ax2.grid(alpha=0.3, linestyle='--')

        # Only display as many timestamps as are needed for the plot
        timeStep = " Default"
        if (len(timestamps) > 10000):
            timeStep = " (960 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::960])
        elif (len(timestamps) > 5000):
            timeStep = " (480 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::480])
        elif (len(timestamps) > 2000):
            timeStep = " (240 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::240])
        elif (len(timestamps) > 800):
            timeStep = " (120 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::120])
        elif (len(timestamps) > 400):
            timeStep = " (60 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::60])
        elif (len(timestamps) > 250):
            timeStep = " (40 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::40])
        elif (len(timestamps) > 100):
            timeStep = " (20 step, # of data rows: " + str(len(timestamps)) + ")"
            plt.xticks(timestamps[::20])
        elif (len(timestamps) > 50):
            timeStep = " (10 step), timestamplength: " + str(len(timestamps))
            plt.xticks(timestamps[::10])
        else:
            timeStep = " (5 step), timestamplength: " + str(len(timestamps))
            plt.xticks(timestamps[::5])

        # Create a separate subplot for the horizontal bar
        ax3 = plt.subplot(gs[2], sharex=ax1)

        # Keep track of which colors have been added to the legend already
        hasColourLegend = [False, False, False, False]

        # Iterate over timestamps and change color based on status
        legend_handles = []  # For custom legend
        for k, timestamp in enumerate(timestamps[:-1]):
            if status[k] == 'Connected':
                bar_color = 'blue'
                if not hasColourLegend[0]:
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=bar_color, label='Connected'))
                hasColourLegend[0] = True
            elif status[k] == 'Charging':
                bar_color = 'green'
                if not hasColourLegend[1]:
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=bar_color, label='Charging'))
                hasColourLegend[1] = True
            elif status[k] == 'None':
                bar_color = 'red'
                if not hasColourLegend[2]:
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=bar_color, label='None'))
                hasColourLegend[2] = True
            else:
                bar_color = 'orange'
                if not hasColourLegend[3]:
                    legend_handles.append(
                        Line2D([0], [0], marker='s', color=bar_color, label='Offline'))
                hasColourLegend[3] = True

            # Add a horizontal bar
            ax3.axhspan(-0.5, 0.5, xmin=(k / len(timestamps)), xmax=((k + 1) / len(timestamps)),
                        facecolor=bar_color, alpha=1)

        # Set the title of the graph
        plt.suptitle(plot_name)

        # Set the title for ax3
        ax3.set_title('Connectivity')

        # Hide y-axis tick labels on ax3
        ax3.set_yticklabels([])
        ax3.set_xlabel('Time' + timeStep)

        # Add small legend on the side of ax3
        fig.legend(handles=legend_handles, loc='lower right')

        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add date and time information as text annotation
        plt.annotate(f"Created: {current_datetime}", xy=(10, 10), xycoords="figure pixels", fontsize=8, color='dimgray')

        # Save the figure
        plt.savefig(os.path.join(results_dir, plot_name))
        plt.close()

# Identify the column index for the "ID" field (adjust as needed)
id_column_index = 9

# Iterate over the data and create plots for each unique ID
current_id = None
current_data = []

for row in data:
    if current_id is None:
        current_id = row[id_column_index]
        current_data.append(row)
    elif row[id_column_index] != current_id:
        # Create a plot for the current ID
        create_plot(current_id, current_data)

        # Update current_id and reset current_data for the new ID
        current_id = row[id_column_index]
        current_data = [row]
    else:
        current_data.append(row)

# Create the last plot for the final ID
create_plot(current_id, current_data)

print("All figures saved to " + results_dir, end='\n')
# Show the plot
plt.show()