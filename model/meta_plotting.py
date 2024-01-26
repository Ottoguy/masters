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

# Set the threshold for disregarding EVs
threshold = 15000  # Replace with your desired threshold

# Filter out EVs with values over the threshold
filtered_meta_df = meta_df[meta_df['Rows'] <= threshold]
disregarded_count = len(meta_df) - len(filtered_meta_df)

# Extract the 'Rows' and 'Type' columns from the filtered DataFrame
connection_durations_1phase = filtered_meta_df[filtered_meta_df['Current_Type'] == '1-Phase']['Rows']
connection_durations_3phase = filtered_meta_df[filtered_meta_df['Current_Type'] == '3-Phase']['Rows']

# Set the bins and range for both histograms
bins = 50
range_vals = (min(connection_durations_1phase.min(), connection_durations_3phase.min()), max(connection_durations_1phase.max(), connection_durations_3phase.max()))

# Plot the distribution with the bottom of bars as green for 1-Phase and blue for 3-Phase
plt.hist(connection_durations_3phase, bins=bins, range=range_vals, color='blue', edgecolor='black', alpha=1, label='3-Phase')
plt.hist(connection_durations_1phase, bins=bins, range=range_vals, color='green', bottom=0, edgecolor='black', alpha=1, label='1-Phase')

plt.xlabel('Connection Duration (Half Minutes)')
plt.ylabel('Number of EVs')
plt.title('Distribution of EV Connection Durations')

# Add legend for disregarded EVs and types
legend_text = f'Disregarded EVs (>{threshold}): {disregarded_count}'
plt.legend([legend_text, '3-Phase', '1-Phase'])

# Adjust the grid with a more subtle appearance
plt.grid(True, which='both', linestyle=':', linewidth=0.3, color='gray', alpha=0.5)
plt.minorticks_on()

plt.show()
