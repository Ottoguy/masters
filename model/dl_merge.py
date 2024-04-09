import os
import glob
import pandas as pd
from datetime import datetime

def DLMerge():
    # Specify the directory where your files are located
    load_path = 'prints/dl_overview/'

    # Create an empty DataFrame with desired columns
    columns = ['RMSE_Clusters', 'RMSE_Intermediate', 'RMSE_Immediate', 'RMSE_Barebones', 'TS_Samples', 'Clusters', 'Clustering Settings',
               'Epochs', 'Batch_Size', 'Layer1_Units', 'Layer2_Units', 'Layer3_Units',
               'Layer1Activation', 'Layer2Activation', 'Layer3Activation', 'Dropout_Rate',
               'ExcludedFeature', 'ShouldEmbed', 'Timestamp', 'MAE_Clusters', 'MAE_Intermediate', 'MAE_Immediate', 'MAE_Barebones', ]
    df = pd.DataFrame(columns=columns)

    # Create a pattern to match CSV files
    file_pattern = '*.csv'

    # Get a list of all CSV files matching the pattern
    file_list = glob.glob(os.path.join(load_path, file_pattern))

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

    #Remove the columns ,timestamp,ts_samples,clusters,test_size,epochs,batch_size,layer1_units,layer2_units,dropout_rate,feature_to_exclude,layer1activation,layer2activation
    df = df.drop(columns=['timestamp','ts_samples','clusters','test_size','epochs','batch_size','layer1_units','layer2_units','dropout_rate','feature_to_exclude','layer1activation','layer2activation', 'OneHotEncode'])

    #Remove the suffix "_DL" from the column names it is present in, and merge those columns with the same name
    df.columns = df.columns.str.replace('_DL', '')
    df = df.groupby(level=0, axis=1).first()

    #Sort the columns like this 'MAE_Clusters', 'MAE_Intermediate', 'MAE_Immediate', 'MAE_Barebones','RMSE_Clusters', 'RMSE_Intermediate', 'RMSE_Immediate', 'RMSE_Barebones','Clusters', 'Epochs', 'Batch_Size', 'Layer1_Units', 'Layer2_Units', 'Layer3_Units','Layer1Activation', 'Layer2Activation', 'Layer3Activation', 'Dropout_Rate','ExcludedFeature', 'ShouldEmbed'
    df = df[['RMSE_Clusters', 'RMSE_Intermediate', 'RMSE_Immediate', 'RMSE_Barebones', 'TS_Samples', 'Clusters', 'Clustering Settings',
                'Epochs', 'Batch_Size', 'Layer1_Units', 'Layer2_Units', 'Layer3_Units',
                'Layer1Activation', 'Layer2Activation', 'Layer3Activation', 'Dropout_Rate',
                'ExcludedFeature', 'ShouldEmbed', 'Timestamp', 'MAE_Clusters', 'MAE_Intermediate', 'MAE_Immediate', 'MAE_Barebones', ]]
    
    #Fill the empty values with None
    df = df.fillna('None')

    #Convert RMSE and MAE columns to numeric
    df['RMSE_Clusters'] = pd.to_numeric(df['RMSE_Clusters'], errors='coerce')

    # Sort by 'RMSE_Clusters'
    df.sort_values(by='RMSE_Clusters', inplace=True)

    output_folder = 'prints/dl_merge/'
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

    return df
