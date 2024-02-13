import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
folder_path = 'prints/meta/'
file_pattern = '*'
file_list = glob.glob(os.path.join(folder_path, file_pattern))
file_list.sort(key=os.path.getmtime, reverse=True)
latest_file = file_list[0]
meta_df = pd.read_csv(latest_file)

# Separate values above 3 and exactly 0
above_3 = meta_df[meta_df['AverageCurrentDifference'] > 3]
exactly_0 = meta_df[meta_df['AverageCurrentDifference'] == 0]

# Calculate additional evenly spaced values
num_additional_bars = 8
additional_values = np.linspace(0, 3, num_additional_bars)

# Create a bar chart
plt.figure(figsize=(12, 6))

# Bar for values <= 0
plt.bar(['<= 0'], [len(exactly_0)], color='skyblue', edgecolor='black')

# Additional evenly spaced bars
for i in range(num_additional_bars):
    val_start = i * 0.375
    val_end = (i + 1) * 0.375

    # Exclude the upper limit for the last bar
    if i == num_additional_bars - 1:
        subset = meta_df[(meta_df['AverageCurrentDifference'] > val_start) & (meta_df['AverageCurrentDifference'] <= val_end)]
    else:
        subset = meta_df[(meta_df['AverageCurrentDifference'] >= val_start) & (meta_df['AverageCurrentDifference'] < val_end)]

    plt.bar([f'{val_start:.2f}-{val_end:.2f}'], [len(subset)], color='skyblue', edgecolor='black')

# Bar for values > 3
plt.bar(['> 3'], [len(above_3)], color='skyblue', edgecolor='black')

plt.xlabel('Average Current Difference')
plt.ylabel('Count')
plt.title('Distribution of Average Current Differences')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the figure with the current date and time in the filename
results_dir = "plots/current_difference"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))