import pandas as pd
import os
import glob
from prince import FAMD
import matplotlib.pyplot as plt
import warnings
import os
import sys
import inspect

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

# Group by 'ID' and aggregate numerical and categorical data as lists
grouped_data = data.groupby('ID').agg(lambda x: x.tolist())

# Separate numerical and categorical columns
numerical_cols = ['Timestamp_sin', 'Timestamp_cos', 'Phase1Current', 'Phase2Current', 'Phase3Current',
                  'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']
categorical_cols = ['ChargingStatus', "ChargingPoint"]

print(grouped_data[numerical_cols])
# Apply FAMD on grouped data
famd = FAMD(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"
)
print("Fitting FAMD...")
famd.fit(grouped_data)

# Print FAMD results
print("FAMD Results:")
print("Eigenvalues:")
print(famd.eigenvalues_summary)
print("Row coordinates:")
print(famd.row_coordinates(grouped_data).head())
print("Column coordinates:")
print(famd.column_coordinates_)
print("Row contributions:")
print(
    famd.row_contributions_
    .sort_values(0, ascending=False)
    .head(5)
    .style.format('{:.3%}')
)
print("Column contributions:")
print(famd.column_contributions_.style.format('{:.0%}'))

# Plot
famd.plot(
    grouped_data,
    x_component=0,
    y_component=1
)

# Get the principal components
principal_components = famd.transform(grouped_data[numerical_cols + categorical_cols])

export_csv_for_id(principal_components, "famd")