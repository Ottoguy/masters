import pandas as pd
import os
import glob
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import itertools
import sys
import inspect
from datetime import datetime

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import functions
from functions import export_csv_for_id

warnings.filterwarnings("ignore")

# Specify the directory where your files are located
data_folder_path = 'prints/all/'
meta_folder_path = 'prints/meta/'

# Create a pattern to match files in the specified format
file_pattern = '*'

# Get a list of all files matching the pattern for both data and meta folders
data_file_list = glob.glob(os.path.join(data_folder_path, file_pattern))
meta_file_list = glob.glob(os.path.join(meta_folder_path, file_pattern))

# Sort the files based on modification time (latest first) for both data and meta folders
data_file_list.sort(key=os.path.getmtime, reverse=True)
meta_file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file for both data and meta folders
latest_data_file = data_file_list[0]
latest_meta_file = meta_file_list[0]

# Load data from the latest data file
data = pd.read_csv(latest_data_file)

# Separate numerical and categorical columns
numerical_cols = ['Timestamp_sin', 'Timestamp_cos', 'Phase1Current', 'Phase2Current', 'Phase3Current',
                  'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']
categorical_cols = ['ChargingStatus', "ChargingPoint"]

# Drop non-numeric columns for PCA
numerical_data = data[numerical_cols]

# Standardize the numerical data
numerical_data_standardized = (numerical_data - numerical_data.mean()) / numerical_data.std()

# Specify the desired number of components
num_components = 4

# Create a subplot with 2 rows and 3 columns
fig, axes = plt.subplots(2, 4, figsize=(15, 10))

# Initialize PCA for the explained variance ratio subplot
pca_variance = PCA()
pca_variance.fit_transform(numerical_data_standardized)

# Plot the explained variance ratio in the first subplot
axes[-1, -1].plot(range(1, len(numerical_cols) + 1), pca_variance.explained_variance_ratio_.cumsum(), marker='o', linestyle='--')
axes[-1, -1].set_title('Explained Variance Ratio')
axes[-1, -1].set_xlabel('Number of Components')
axes[-1, -1].set_ylabel('Cumulative Explained Variance Ratio')

# Initialize PCA for the desired number of components
pca = PCA(n_components=num_components)
principal_components = pca.fit_transform(numerical_data_standardized)
columns = [f'PC{i}' for i in range(1, num_components + 1)]
principal_df = pd.DataFrame(data=principal_components, columns=columns)

# Create combinations of principal components for scatter plots
combinations = list(itertools.combinations(range(num_components), 2))

# Loop through combinations and create scatter plots
for i, (pc1, pc2) in enumerate(combinations):
    print(f'Creating PCA for components {pc1+1} and {pc2+1}...')
    row = i // 3
    col = i % 3
    axes[row, col].scatter(principal_df[f'PC{pc1+1}'], principal_df[f'PC{pc2+1}'], c='b', marker='o', alpha=0.5)
    axes[row, col].set_title(f'PCA - PC{pc1+1} vs PC{pc2+1}')
    axes[row, col].set_xlabel(f'Principal Component {pc1+1}')
    axes[row, col].set_ylabel(f'Principal Component {pc2+1}')

# Adjust layout and save the figure with the current date and time in the filename
plt.tight_layout()
results_dir = "plots/pre_clustering/pca/"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(os.path.join(results_dir, current_datetime + '.png'))
export_csv_for_id(principal_df, "pca")
print("PCA results and plot saved")