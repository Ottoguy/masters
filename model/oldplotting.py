from load_dans import all_data as dans_data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from matplotlib.lines import Line2D
from os import path
from datetime import datetime

# Choose which data to use
data = dans_data

# Choose preprocessing type
# preprocessing_type = "none"
# preprocessing_type = "diff"
# preprocessing_type = "mean"
# preprocessing_type = "std"

while True:
    print('Enter preferred preprocessing (none, diff, mean, or std):')
    preprocessing_type = input()

    # Find script directory
    script_dir = os.path.dirname(__file__)

    if preprocessing_type == "none":
        results_dir = os.path.join(script_dir, 'Preprocessing_figures/none/')
        break
    elif preprocessing_type == "diff":
        results_dir = os.path.join(script_dir, 'Preprocessing_figures/diff/')
        break
    elif preprocessing_type == "mean":
        results_dir = os.path.join(script_dir, 'Preprocessing_figures/mean/')
        break
    elif preprocessing_type == "std":
        results_dir = os.path.join(script_dir, 'Preprocessing_figures/std/')
        break
    else:
        print("Invalid preprocessing type, please enter a valid data type.")

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)

# Iterate over all files
for i in range(1, len(data)):
    # Get the data for the current file (doc is a csv file)
    doc = data[i]

    if path.exists(results_dir + "/" + doc[0][0] + "_" + preprocessing_type + ".png"):
        print("File " + doc[0][0] + "_" + preprocessing_type +
              ".png already exists, skipping...", end='\r')
    else:
        # Extract relevant data for plotting
        timestamps = []
        status = []  # Added for Online/Offline status

        if preprocessing_type == "none":
            label_upper = "Effect (kW)"
            label_lower = "Voltage (V)"
            phase1_effect = []
            phase2_effect = []
            phase3_effect = []
            phase1_voltage = []
            phase2_voltage = []
            phase3_voltage = []
        elif preprocessing_type == "diff":
            label_upper = "Effect Diff (kW)"
            label_lower = "Voltage Diff (V)"
            effect_12_diff = []
            effect_23_diff = []
            effect_13_diff = []
            voltage_12_diff = []
            voltage_23_diff = []
            voltage_13_diff = []
        elif preprocessing_type == "mean":
            label_upper = "Effect Mean (kW)"
            label_lower = "Voltage Mean (V)"
            effect_mean = []
            effect_12_mean = []
            effect_23_mean = []
            effect_13_mean = []
            voltage_mean = []
            voltage_12_mean = []
            voltage_23_mean = []
            voltage_13_mean = []
        elif preprocessing_type == "std":
            label_upper = "Effect STD"
            label_lower = "Voltage STD"
            effect_std = []
            effect_12_std = []
            effect_23_std = []
            effect_13_std = []
            voltage_std = []
            voltage_12_std = []
            voltage_23_std = []
            voltage_13_std = []

        for row in doc:
            # Extract time without the first 11 and last 7 characters (Only the time left)
            time_str = row[1][11:-7]
            timestamps.append(time_str)
            status.append(row[-3])

            if preprocessing_type == "none":
                phase1_effect.append(float(row[3]))
                phase2_effect.append(float(row[4]))
                phase3_effect.append(float(row[5]))
                phase1_voltage.append(float(row[6]))
                phase2_voltage.append(float(row[7]))
                phase3_voltage.append(float(row[8]))
            elif preprocessing_type == "diff":
                effect_12_diff.append(abs(float(row[3]) - float(row[4])))
                effect_23_diff.append(abs(float(row[4]) - float(row[5])))
                effect_13_diff.append(abs(float(row[3]) - float(row[5])))
                voltage_12_diff.append(abs(float(row[6]) - float(row[7])))
                voltage_23_diff.append(abs(float(row[7]) - float(row[8])))
                voltage_13_diff.append(abs(float(row[6]) - float(row[8])))
            elif preprocessing_type == "mean":
                effect_mean.append(
                    (float(row[3]) + float(row[4]) + float(row[5]))/3)
                effect_12_mean.append((float(row[3]) + float(row[4]))/2)
                effect_23_mean.append((float(row[4]) + float(row[5]))/2)
                effect_13_mean.append((float(row[3]) + float(row[5]))/2)
                voltage_mean.append(
                    (float(row[6]) + float(row[7]) + float(row[8]))/3)
                voltage_12_mean.append((float(row[6]) + float(row[7]))/2)
                voltage_23_mean.append((float(row[7]) + float(row[8]))/2)
                voltage_13_mean.append((float(row[6]) + float(row[8]))/2)
            elif preprocessing_type == "std":
                effect_std.append(
                    np.std([float(row[3]), float(row[4]), float(row[5])]))
                effect_12_std.append(np.std([float(row[3]), float(row[4])]))
                effect_23_std.append(np.std([float(row[4]), float(row[5])]))
                effect_13_std.append(np.std([float(row[3]), float(row[5])]))
                voltage_std.append(
                    np.std([float(row[6]), float(row[7]), float(row[8])]))
                voltage_12_std.append(np.std([float(row[6]), float(row[7])]))
                voltage_23_std.append(np.std([float(row[7]), float(row[8])]))
                voltage_13_std.append(np.std([float(row[6]), float(row[8])]))

        if preprocessing_type == "none":
            upper1 = phase1_effect
            upper2 = phase2_effect
            upper3 = phase3_effect
            lower1 = phase1_voltage
            lower2 = phase2_voltage
            lower3 = phase3_voltage
        elif preprocessing_type == "diff":
            upper1 = effect_12_diff
            upper2 = effect_23_diff
            upper3 = effect_13_diff
            lower1 = voltage_12_diff
            lower2 = voltage_23_diff
            lower3 = voltage_13_diff
        elif preprocessing_type == "mean":
            upper1 = effect_mean
            upper2 = effect_12_mean
            upper3 = effect_23_mean
            upper4 = effect_13_mean
            lower1 = voltage_mean
            lower2 = voltage_12_mean
            lower3 = voltage_23_mean
            lower4 = voltage_13_mean
        elif preprocessing_type == "std":
            upper1 = effect_std
            upper2 = effect_12_std
            upper3 = effect_23_std
            upper4 = effect_13_std
            lower1 = voltage_std
            lower2 = voltage_12_std
            lower3 = voltage_23_std
            lower4 = voltage_13_std

        # Create subplots using gridspec
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[
            8, 8, 1], hspace=0.4, wspace=0.0)

        # Plot the effects in the first subplot
        ax1 = plt.subplot(gs[0])
        ax1.plot(timestamps, upper1,
                 label=label_upper + " 1", linestyle='--')
        ax1.plot(timestamps, upper2,
                 label=label_upper + " 2", linestyle='-.')
        ax1.plot(timestamps, upper3,
                 label=label_upper + " 3", linestyle=':')
        if preprocessing_type == "mean" or preprocessing_type == "std":
            ax1.plot(timestamps, upper4,
                     label=label_upper + " (all)", linestyle=':')
        ax1.set_ylabel('Effect Values')
        ax1.legend()
        ax1.margins(x=0)

        # Plot the voltages in the second subplot
        ax2 = plt.subplot(gs[1], sharex=ax1)
        ax2.plot(timestamps, lower1,
                 label=label_lower + " 1", linestyle='--')
        ax2.plot(timestamps, lower2,
                 label=label_lower + " 2", linestyle='-.')
        ax2.plot(timestamps, lower3,
                 label=label_lower + " 3", linestyle=':')
        if preprocessing_type == "mean" or preprocessing_type == "std":
            ax2.plot(timestamps, lower4,
                     label=label_lower + " (all)", linestyle=':')
        ax2.set_ylabel('Voltage Values')
        ax2.legend()
        ax2.margins(x=0)

        # Add unintrusive grid to both subplots
        ax1.grid(alpha=0.3, linestyle='--')
        ax2.grid(alpha=0.3, linestyle='--')

        # Only display as many timestamps as are needed for the plot
        timeStep = " Default"
        if (len(timestamps) > 2000):
            timeStep = " (240 step)"
            plt.xticks(timestamps[::240])
        elif (len(timestamps) > 1000):
            timeStep = " (120 step)"
            plt.xticks(timestamps[::120])
        elif (len(timestamps) > 500):
            timeStep = " (60 step)"
            plt.xticks(timestamps[::60])
        elif (len(timestamps) > 250):
            timeStep = " (40 step)"
            plt.xticks(timestamps[::40])
        elif (len(timestamps) > 100):
            timeStep = " (20 step)"
            plt.xticks(timestamps[::20])
        elif (len(timestamps) > 50):
            timeStep = " (10 step)"
            plt.xticks(timestamps[::10])
        else:
            timeStep = " (5 step)"
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
        plt.suptitle(doc[0][0])

        # Set the title for ax3
        ax3.set_title('Connectivity')

        # Hide y-axis tick labels on ax3
        ax3.set_yticklabels([])
        ax3.set_xlabel('Time' + timeStep)

        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Add date and time information as text annotation
        plt.annotate(f"Created: {current_datetime}", xy=(10, 10),
                     xycoords="figure pixels", fontsize=8, color='dimgray')

        # Add small legend on the side of ax3
        fig.legend(handles=legend_handles, loc='lower right')

        # Save the figure
        print("Saving figure " + str(i) + " of " + str(len(data)) +
              " as " + doc[0][0] + "_" + preprocessing_type + ".png", end='\r')
        plt.savefig(results_dir + doc[0][0] +
                    "_" + preprocessing_type + ".png")
        plt.close()

print("All figures saved to " + results_dir, end='\n')
# Show the plot
plt.show()
