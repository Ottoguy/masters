import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Specify the directory where your files are located
folder_path = 'prints/meta_df/'

# Create a pattern to match files in the specified format
file_pattern = 'meta_df_*'

# Get a list of all files matching the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

# Sort the files based on modification time (latest first)
file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file
latest_file = file_list[0]

# Load your data from the latest file
meta_df = pd.read_csv(latest_file)

# Set the threshold (half-minutes) for disregarding EVs
#5760 is 48hrs
threshold = 5760  # Replace with your desired threshold

# Filter out EVs with values over the threshold
filtered_meta_df = meta_df[meta_df['Rows'] <= threshold]
disregarded_count = len(meta_df) - len(filtered_meta_df)

# Extract the 'Rows' and 'Type' columns from the filtered DataFrame
connection_durations_1phase = filtered_meta_df[filtered_meta_df['Current_Type'] == '1-Phase']['Rows']
connection_durations_3phase = filtered_meta_df[filtered_meta_df['Current_Type'] == '3-Phase']['Rows']

# Convert durations to hours and minutes
connection_durations_1phase_hours_minutes = connection_durations_1phase / 2 / 60
connection_durations_3phase_hours_minutes = connection_durations_3phase / 2 / 60

# Set the bins and range for both histograms in hours and minutes
bins = 50
range_vals_hours_minutes = (
    min(connection_durations_1phase_hours_minutes.min(), connection_durations_3phase_hours_minutes.min()),
    max(connection_durations_1phase_hours_minutes.max(), connection_durations_3phase_hours_minutes.max())
)

# Plot the distribution with the bottom of bars as green for 1-Phase and blue for 3-Phase
plt.hist(connection_durations_3phase_hours_minutes, bins=bins, range=range_vals_hours_minutes, color='blue', edgecolor='black', alpha=1, label='3-Phase')
plt.hist(connection_durations_1phase_hours_minutes, bins=bins, range=range_vals_hours_minutes, color='green', bottom=0, edgecolor='black', alpha=1, label='1-Phase')

plt.xlabel('Connection Duration (Hours)')  # Update x-axis label
plt.ylabel('Number of EVs')
plt.title('Distribution of EV Connection Durations')

# Add legend for disregarded EVs and types
legend_text = f'Omitted EVs (>{threshold/120} hours): {disregarded_count}'
plt.legend([legend_text, '3-Phase', '1-Phase'])

# Adjust the grid with a more subtle appearance
plt.grid(True, which='both', linestyle=':', linewidth=0.3, color='gray', alpha=0.5)
plt.minorticks_on()

plt.show()