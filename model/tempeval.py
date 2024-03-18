import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import re

#Evaluates the data in prints/deep_learning and saves the results in prints/deep_learning/ 
#To be used when deeo learning is aborted or has crashed, but results are to be fetched from the prints/deep_learning files

# Directory where the CSV files are saved
directory = 'prints/deep_learning'

# Initialize an empty list to store DataFrames
results_dfs = []

# Iterate over each file in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        df = pd.read_csv(filepath)
        
        # Extract settings from filename using regular expressions
        match = re.match(r".*?(\d{8}_\d{6})_ts_samples_(\d+)_clusters_(\d+)_test_size_(\d+\.\d+)_epochs_(\d+)_batch_size_(\d+)_l1_units_(\d+)_l2_units_(\d+)_l3_units_(\d+)_dropout_rate_(\d+\.\d+)_exclude_(.*?)_l1activation_(.*?)_l2activation_(.*?)_l3activation_(.*?)\.csv", filename)
        if match:
            settings_dict = {
                'timestamp': match.group(1),
                'ts_samples': int(match.group(2)),
                'clusters': int(match.group(3)),
                'test_size': float(match.group(4)),
                'epochs': int(match.group(5)),
                'batch_size': int(match.group(6)),
                'layer1_units': int(match.group(7)),
                'layer2_units': int(match.group(8)),
                'layer3_units': int(match.group(9)),
                'dropout_rate': float(match.group(10)),
                'feature_to_exclude': match.group(11),
                'layer1activation': match.group(12),
                'layer2activation': match.group(13),
                'layer3activation': match.group(14)
            }
        else:
            print(f"Could not parse settings from filename: {filename}")
            continue
        
        # Calculate RMSE for each model
        rmse_clusters = sqrt(mean_squared_error(df['Real'], df['Clusters']))
        rmse_intermediate = sqrt(mean_squared_error(df['Real'], df['Intermediate']))
        rmse_immediate = sqrt(mean_squared_error(df['Real'], df['Immediate']))
        rmse_barebones = sqrt(mean_squared_error(df['Real'], df['Barebones']))
        
        # Add RMSE values to settings dictionary
        settings_dict.update({
            'RMSE_Clusters': rmse_clusters,
            'RMSE_Intermediate': rmse_intermediate,
            'RMSE_Immediate': rmse_immediate,
            'RMSE_Barebones': rmse_barebones
        })
        
        # Convert settings dictionary to DataFrame and append to list
        results_dfs.append(pd.DataFrame([settings_dict]))

# Concatenate all DataFrames in the list
if results_dfs:
    results_aggregated = pd.concat(results_dfs, ignore_index=True)
    
    # Rearrange columns to have RMSE_Clusters first
    column_order = ['RMSE_Clusters', 'RMSE_Intermediate', 'RMSE_Immediate', 'RMSE_Barebones'] + \
                   [col for col in results_aggregated.columns if col not in ['RMSE_Clusters', 'RMSE_Intermediate', 'RMSE_Immediate', 'RMSE_Barebones']]
    results_aggregated = results_aggregated[column_order]
    
    # Sort DataFrame by RMSE_Clusters
    results_aggregated = results_aggregated.sort_values(by='RMSE_Clusters')
    
    # Save aggregated results to CSV
    results_aggregated.to_csv('prints/deep_learning/aggregated_results3.csv', index=False)
