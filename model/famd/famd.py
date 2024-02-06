import pandas as pd
import os
import glob
from prince import FAMD
import matplotlib.pyplot as plt

# Specify the directory where your files are located
folder_path = 'prints/all/'

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

# Drop columns that are not needed
data = data.drop(columns=['Timestamp', "Filename"], axis=1)

# Separate numerical and categorical columns
numerical_cols = ['Timestamp_sin', 'Timestamp_cos', 'Phase1Current', 'Phase2Current', 'Phase3Current',
                  'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage']
categorical_cols = ['Effect', 'ChargingStatus']

# Perform Factor Analysis of Mixed Data (FAMD)
famd = FAMD(n_components=2, n_iter=3, random_state=42)
famd.fit(data[numerical_cols + categorical_cols])

# Get the principal components
principal_components = famd.transform(data[numerical_cols + categorical_cols])

# Display the principal components
print(principal_components)

# Scatter plot of principal components
plt.scatter(principal_components.iloc[:, 0], principal_components.iloc[:, 1])
plt.title('Scatter Plot of Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()