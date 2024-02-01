import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import glob
import itertools
from datetime import datetime
from scipy.special import comb
import torch

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
df = pd.read_csv(latest_file)

print("CSV file loaded")

# Convert 'TimeConnected' and 'TimeDisconnected' to datetime
df['TimeConnected'] = pd.to_datetime(df['TimeConnected'].copy())
df['TimeDisconnected'] = pd.to_datetime(df['TimeDisconnected'].copy())

print("Extracting features...")

# Extract relevant columns (excluding ID)
features = df[['Half_Minutes', 'Charging_Half_Minutes', 'Energy_Uptake', 'ChargingPoint', 'Current_Type', 'FullyCharged', 'Weekend']].copy()

# Extract hour and minute information from 'TimeConnected' and 'TimeDisconnected'
features['Hour_Connected'] = df['TimeConnected'].dt.hour.copy()
features['Minute_Connected'] = df['TimeConnected'].dt.minute.copy()
features['Hour_Disconnected'] = df['TimeDisconnected'].dt.hour.copy()
features['Minute_Disconnected'] = df['TimeDisconnected'].dt.minute.copy()

# Convert boolean columns to numeric (0 and 1)
features['FullyCharged'] = features['FullyCharged'].astype(int)
features['Weekend'] = features['Weekend'].astype(int)

print("One-hot encoding...")

# One-hot encode 'ChargingPoint' and 'CurrentType'
features = pd.get_dummies(features, columns=['ChargingPoint', 'Current_Type'], prefix=['ChargingPoint', 'CurrentType'])

print("Standardizing...")
# Standardize the data
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

# Create a DataFrame to store the results
results_df = pd.DataFrame(columns=['Components', 'ExplainedVariance', 'CumulativeVariance'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Performing PCA...")
# Iterate over all combinations and perform PCA
for r in range(1, len(features.columns) + 1):
    # Calculate total combinations using binomial coefficient (n choose k)
    total_combinations = comb(len(features.columns), r)
    
    for idx, components in enumerate(itertools.combinations(features.columns, r), 1):
        # Convert the combination to a list for indexing
        components_list = list(components)
        
        # Extract selected features
        selected_features = features[components_list].values.astype('float32')
        selected_features_tensor = torch.from_numpy(selected_features).to(device)

        # Apply PCA
        _, _, v = torch.linalg.svd(selected_features_tensor, full_matrices=False)
        explained_variance_ratio = torch.square(v).sum(dim=1) / torch.square(selected_features_tensor).sum()
        cumulative_variance = explained_variance_ratio.cumsum()[-1].item()

        # Create a new DataFrame for each iteration
        iteration_df = pd.DataFrame({'Components': [components_list], 'ExplainedVariance': [explained_variance_ratio[-1].item()], 'CumulativeVariance': [cumulative_variance]})
        
        # Concatenate the new DataFrame to the results_df
        results_df = pd.concat([results_df, iteration_df], ignore_index=True)

        # Print progress
        progress_percent = (idx / total_combinations) * 100
        print(f"Progress: {idx}/{total_combinations} combinations analyzed ({progress_percent:.2f}%)", end='\r', flush=True)

# Sort the DataFrame by the desired metric (e.g., cumulative variance)
results_df = results_df.sort_values(by='CumulativeVariance', ascending=False)

output_folder = 'prints/pca_pytorch'

# Create a folder named "prints/pca" if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Get the current date and time
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create the file name
output_file = f"{output_folder}/{current_datetime}.csv"

# Print desired_rows to a CSV file
results_df.to_csv(output_file, index=False)