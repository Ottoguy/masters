import pandas as pd
import os
import glob
from prince import FAMD
import matplotlib.pyplot as plt
import warnings

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

# Load meta data from the latest meta file
meta_data = pd.read_csv(latest_meta_file)

# Separate numerical and categorical columns
numerical_cols = ['Timestamp_sin', 'Timestamp_cos', 'Phase1Current', 'Phase2Current', 'Phase3Current',
                  'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']
categorical_cols = ['Effect', 'ChargingStatus']

# Extract required numerical and categorical features from the meta data
numerical_meta_cols = ['Half_Minutes', 'Charging_Half_Minutes', 'TimeConnected_sin', 'TimeConnected_cos', 
                       'TimeDisconnected_sin', 'TimeDisconnected_cos', 'Energy_Uptake']
categorical_meta_cols = ['FullyCharged', 'ChargingPoint', 'Current_Type', 'Weekend']

# Merge data and meta data on the common ID column
merged_data = pd.merge(data, meta_data[['ID'] + numerical_meta_cols + categorical_meta_cols], on='ID')

# Drop columns that are not needed
merged_data = merged_data.drop(columns=['Timestamp', 'Filename'], axis=1)

# Separate numerical and categorical columns
numerical_cols = numerical_cols + numerical_meta_cols
categorical_cols = categorical_cols + categorical_meta_cols


print("Performing FAMD...")
# Perform Factor Analysis of Mixed Data (FAMD)
famd = FAMD(
    n_components=2,
    n_iter=3,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
print("Fitting FAMD...")
famd.fit(merged_data[numerical_cols + categorical_cols])

print("FAMD Results:")
print("Eigenvalues:")
print(famd.eigenvalues_summary)
print("Row coordinates:")
print(famd.row_coordinates(merged_data).head())
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
    merged_data,
    x_component=0,
    y_component=1
)

print("Principal components:")
# Get the principal components
principal_components = famd.transform(merged_data[numerical_cols + categorical_cols])

# Scatter plot of principal components
plt.scatter(principal_components.iloc[:, 0], principal_components.iloc[:, 1])
plt.title('Scatter Plot of Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

plt.plot(range(1, len(famd.explained_inertia_) + 1), famd.explained_inertia_, marker='o')
plt.title('Scree Plot')
plt.xlabel('Component Number')
plt.ylabel('Explained Inertia')
plt.show()