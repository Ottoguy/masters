import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime

def TSMerge(columns):
    # Specify the directory where your files are located
    load_path = 'prints/ts_eval_experimental/'
    # Create a pattern to match CSV files
    file_pattern = '*.csv'
    # Get a list of all CSV files matching the pattern
    file_list = glob.glob(os.path.join(load_path, file_pattern))
    #Initiate df with columns from columns
    df = pd.DataFrame(columns=columns)
    # Iterate over each file, load it, and concatenate to the main DataFrame
    for file in file_list:
        temp_df = pd.read_csv(file)
        # Fill missing columns with None
        for column in columns:
            if column not in temp_df.columns:
                temp_df[column] = None
        #Add the file name to the DataFrame
        temp_df['Timestamp'] = os.path.basename(file).split('.')[0]
        df = pd.concat([df, temp_df])
    #Remove columns with only None values
    df = df.dropna(axis=1, how='all')

    #Sort the dataframe by 'SilhouetteScore' and make this feature the first column
    df = df.sort_values(by='SilhouetteScore', ascending=False)
    df.insert(0, 'Silhouette_Score', df['SilhouetteScore'])
    df = df.drop(columns=['SilhouetteScore'])
    #Rename 'Silhouette_Score' to 'SilhouetteScore'
    df = df.rename(columns={'Silhouette_Score': 'SilhouetteScore'})

    output_folder = 'prints/ts_merge/'
    # Create a folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Create the file name
    print(f"Creating the file {current_datetime}.csv")
    output_file = f"{output_folder}/{current_datetime}.csv"
    # Print desired_rows to a CSV file
    df.to_csv(output_file, index=False)
    #Print path to the created file
    print(f"Results saved to {output_file}")