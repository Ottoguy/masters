import matplotlib.pyplot as plt
from matplotlib import gridspec
from datetime import datetime

# from load_xlsx import data as xlsx_data
# from load_ångan import data as ångan_data
from load_dans import all_data as dans_data

# Choose which data to use
data = dans_data
print(len(data))

# Choosing which files to use
doc1 = data[1]
print(doc1[0])
doc2 = data[22]
print(doc2[0])

# Extract relevant data for plotting
timestamps = []
status = []  # Added for Online/Offline status
phase1_effect = []
phase2_effect = []
phase3_effect = []
phase1_voltage = []
phase2_voltage = []
phase3_voltage = []

for row in doc2:
    # Extract time without the first 11 and last 7 characters
    time_str = row[1][11:-7]
    timestamps.append(time_str)
    status.append(row[-3])
    phase1_effect.append(float(row[3]))
    phase2_effect.append(float(row[4]))
    phase3_effect.append(float(row[5]))
    phase1_voltage.append(float(row[6]))
    phase2_voltage.append(float(row[7]))
    phase3_voltage.append(float(row[8]))

# Create subplots using gridspec
fig = plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(3, 1, height_ratios=[8, 8, 1], hspace=0.4)

# Plot the effects in the first subplot
ax1 = plt.subplot(gs[0])
ax1.plot(timestamps, phase1_effect, label='Phase 1 effect')
ax1.plot(timestamps, phase2_effect, label='Phase 2 effect')
ax1.plot(timestamps, phase3_effect, label='Phase 3 effect')
ax1.set_ylabel('Effect Values')
ax1.legend()

# Plot the voltages in the second subplot
ax2 = plt.subplot(gs[1], sharex=ax1)
ax2.plot(timestamps, phase1_voltage, label='Phase 1 voltage')
ax2.plot(timestamps, phase2_voltage, label='Phase 2 voltage')
ax2.plot(timestamps, phase3_voltage, label='Phase 3 voltage')
ax2.set_xlabel('Time')
ax2.set_ylabel('Voltage Values')
ax2.legend()


# Add unintrusive grid to both subplots
ax1.grid(alpha=0.3, linestyle='--')
ax2.grid(alpha=0.3, linestyle='--')

# Display every 10th timestamp on the x-axis
plt.xticks(timestamps[::20])

# Create a separate subplot for the horizontal bar
ax3 = plt.subplot(gs[2], sharex=ax1)

# Iterate over timestamps and change color based on status
for i, timestamp in enumerate(timestamps[:-1]):
    if status[i] == 'Connected':
        bar_color = 'blue'
    elif status[i] == 'Charging':
        bar_color = 'green'
    elif status[i] == 'None':
        bar_color = 'red'
    else:
        bar_color = 'orange'

    # Add a horizontal bar
    ax3.axhspan(-0.5, 0.5, xmin=i / len(timestamps), xmax=(i + 1) / len(timestamps),
                facecolor=bar_color, alpha=0.8)

# Set the title of the graph
plt.suptitle(doc2[0][0])

# Set the title for ax3
ax3.set_title('Connectivity')

# Hide y-axis and x-axis tick labels on ax3
ax3.set_yticklabels([])
ax3.set_xticklabels([])

# Show the plot
plt.show()
