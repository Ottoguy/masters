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

# Separate values above 2 and exactly 0
above_2 = meta_df[meta_df['AverageVoltageDifference'] > 2]
exactly_0 = meta_df[meta_df['AverageVoltageDifference'] == 0]

# Calculate additional evenly spaced values
num_additional_bars = 8
additional_values = np.linspace(0, 2, num_additional_bars)

# Create a bar chart
plt.figure(figsize=(12, 6))

# Bar for values <= 0
plt.bar(['<= 0'], [len(exactly_0)], color='skyblue', edgecolor='black')

# Additional evenly spaced bars
for val in additional_values:
    subset = meta_df[(meta_df['AverageVoltageDifference'] > val) & (meta_df['AverageVoltageDifference'] <= val + 0.25)]
    plt.bar([f'{val:.2f}-{val + 0.25:.2f}'], [len(subset)], color='skyblue', edgecolor='black')

# Bar for values > 2
plt.bar(['> 2'], [len(above_2)], color='skyblue', edgecolor='black')

plt.xlabel('Average Voltage Difference')
plt.ylabel('Count')
plt.title('Distribution of Average Voltage Differences')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Display the plot
plt.show()
