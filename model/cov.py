import pandas as pd
import os
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
data = pd.read_csv(latest_file)

# Convert 'TimeConnected' and 'TimeDisconnected' to datetime objects
data['TimeConnected'] = pd.to_datetime(data['TimeConnected'])
data['TimeDisconnected'] = pd.to_datetime(data['TimeDisconnected'])

# Extract hours and minutes
data['TimeConnected'] = data['TimeConnected'].dt.hour * 60 + data['TimeConnected'].dt.minute
data['TimeDisconnected'] = data['TimeDisconnected'].dt.hour * 60 + data['TimeDisconnected'].dt.minute

# Convert 'TimeConnected' and 'TimeDisconnected' to sine and cosine values
data['TimeConnected_sin'] = np.sin(2 * np.pi * data['TimeConnected'] / (24 * 60))
data['TimeConnected_cos'] = np.cos(2 * np.pi * data['TimeConnected'] / (24 * 60))

data['TimeDisconnected_sin'] = np.sin(2 * np.pi * data['TimeDisconnected'] / (24 * 60))
data['TimeDisconnected_cos'] = np.cos(2 * np.pi * data['TimeDisconnected'] / (24 * 60))

# Exclude the 'ID' field and original time columns
numeric_columns = data.drop(columns=['ID', 'TimeConnected', 'TimeDisconnected']).select_dtypes(include=['float64', 'int64'])

# Create covariance matrix
covariance_matrix = numeric_columns.cov()

# Set up the matplotlib figure
plt.figure(figsize=(14, 12))

# Create a heatmap with a diverging colormap
sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, fmt=".2f")

# Display the plot
plt.title('Covariance Matrix Heatmap')
plt.show()
