import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
from matplotlib.lines import Line2D
from datetime import datetime
import pandas as pd
import glob

# Specify the directory where your files are located
folder_path = 'prints/df/'

# Create a pattern to match files in the specified format
file_pattern = 'df_*'

# Get a list of all files matching the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

# Sort the files based on modification time (latest first)
file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file
latest_file = file_list[0]

# Load your df from the latest file
df = pd.read_csv(latest_file)

print(df.iloc[0])

# Extract relevant df for plotting
timestamps = []
status = []  # Added for Online/Offline status

phase1_effect = []
phase2_effect = []
phase3_effect = []
phase1_voltage = []
phase2_voltage = []
phase3_voltage = []

print(df)

for row in df:
    # Extract time without the first 11 and last 7 characters (Only the time left)
    time_str = row[1][11:-7]
    timestamps.append(time_str)
    status.append(row[-3])

    phase1_effect.append(float(row[3]))
    phase2_effect.append(float(row[4]))
    phase3_effect.append(float(row[5]))
    phase1_voltage.append(float(row[6]))
    phase2_voltage.append(float(row[7]))
    phase3_voltage.append(float(row[8]))
    
    upper1 = phase1_effect
    upper2 = phase2_effect
    upper3 = phase3_effect
    lower1 = phase1_voltage
    lower2 = phase2_voltage
    lower3 = phase3_voltage

# Create subplots using gridspec
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[
    8, 8, 1], hspace=0.4, wspace=0.0)

# Plot the effects in the first subplot
ax1 = plt.subplot(gs[0])
ax1.plot(timestamps, upper1,
            label="Phase 1 Effect (kW)", linestyle='--')
ax1.plot(timestamps, upper2,
            label="Phase 2 Effect (kW)", linestyle='-.')
ax1.plot(timestamps, upper3,
            label="Phase 3 Effect (kW)", linestyle=':')

ax1.set_ylabel('Effect Values')
ax1.legend()
ax1.margins(x=0)

# Plot the voltages in the second subplot
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(timestamps, lower1,
            label="Phase 1 Voltage (V)", linestyle='--')
ax2.plot(timestamps, lower2,
            label="Phase 2 Voltage (V)", linestyle='-.')
ax2.plot(timestamps, lower3,
            label="Phase 3 Voltage (V)", linestyle=':')

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
plt.suptitle(df[0][0])

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

for i in enumerate(df):
    # Save the figure
    print("Saving figure " + str(i) + " of " + str(len(df)) +
        " as " + df[0][0] + ".png", end='\r')
    plt.savefig(results_dir + df[0][0] +
            ".png")
    plt.close()

print("All figures saved to " + results_dir, end='\n')
# Show the plot
plt.show()
