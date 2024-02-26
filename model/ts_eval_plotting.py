import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Load data
folder_path = 'prints/ts_eval/'
file_pattern = '*'
file_list = glob.glob(os.path.join(folder_path, file_pattern))
file_list.sort(key=os.path.getmtime, reverse=True)
latest_file = file_list[0]
data = pd.read_csv(latest_file)

# Separate data for 1-phase and 3-phase
data_1_phase = data[['Num Clusters', 'Average Distance 1-Phase', 'Average Charging Distance 1-Phase', 'Silhouette Score 1-Phase']]
data_3_phase = data[['Num Clusters', 'Average Distance 3-Phase', 'Average Charging Distance 3-Phase', 'Silhouette Score 3-Phase']]

# Plotting
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

# Plot for 1-phase
axes[0].plot(data_1_phase['Num Clusters'], data_1_phase['Average Distance 1-Phase'], label='Avg Distance 1-Phase')
axes[0].plot(data_1_phase['Num Clusters'], data_1_phase['Average Charging Distance 1-Phase'], label='Avg Charging Distance 1-Phase')
axes[0].set_title('1-Phase Data')
axes[0].set_xlabel('Number of Clusters')
axes[0].set_ylabel('Distance')
axes[0].legend()

# Plot for 3-phase
axes[1].plot(data_3_phase['Num Clusters'], data_3_phase['Average Distance 3-Phase'], label='Avg Distance 3-Phase')
axes[1].plot(data_3_phase['Num Clusters'], data_3_phase['Average Charging Distance 3-Phase'], label='Avg Charging Distance 3-Phase')
axes[1].set_title('3-Phase Data')
axes[1].set_xlabel('Number of Clusters')
axes[1].set_ylabel('Distance')
axes[1].legend()

# Save the figure with the current date and time in the filename
results_dir = "plots/ts_eval"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '_subplots.png'))
plt.show()
