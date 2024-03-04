import os
import glob
import pandas as pd
from functions import export_csv_for_id

def PreprocSplit(get_immediate_features, get_intermediate_features, get_final_features):
    # Specify the directory where your files are located
    folder_path = 'prints/all/'
    meta_path = 'prints/meta/'

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

    meta_list = glob.glob(os.path.join(meta_path, file_pattern))
    meta_list.sort(key=os.path.getmtime, reverse=True)
    latest_meta = meta_list[0]
    meta_df = pd.read_csv(latest_meta)

    if get_immediate_features:
        # Add immediate features from meta_df (ID,TimeConnected,ChargingPoint,Current_Type,Weekend,TimeConnected_sin,TimeConnected_cos) to a new dataframe
        meta_df_immediate_features = meta_df[['ID', 'TimeConnected', 'ChargingPoint', 'Weekend', 'TimeConnected_sin',
                                        'TimeConnected_cos']].copy()
        export_csv_for_id(meta_df_immediate_features, "preproc_immediate")

    if get_intermediate_features:
        # Add intermediate features from meta_df (ID, MaxVoltage, MaxCurrent, FullyCharged, Energy_Uptake, AverageVoltageDifference, AverageCurrentDifference, TS_Cluster) to a new dataframe
        meta_df_intermediate_features = meta_df[['ID', 'MaxVoltage', 'MaxCurrent', 'FullyCharged', 'Current_Type', 'Energy_Uptake', 'AverageVoltageDifference',
                                            'AverageCurrentDifference']].copy()
        export_csv_for_id(meta_df_intermediate_features, "preproc_intermediate")

    if get_final_features:
        #Add final deatures from meta_df (ID, Half_Minutes,Charging_Half_Minutes,TimeDisconnected,MaxVoltage,MaxCurrent,FullyCharged,Energy_Uptake,sin,TimeDisconnected_sin,TimeDisconnected_cos,AverageVoltageDifference,AverageCurrentDifference) to a new dataframe
        meta_df_final_features = meta_df[['ID', 'Half_Minutes', 'Charging_Half_Minutes', 'TimeDisconnected', 'MaxVoltage', 'MaxCurrent',
                                    'FullyCharged', 'Energy_Uptake', 'TimeDisconnected_sin', 'TimeDisconnected_cos', 'AverageVoltageDifference',
                                    'AverageCurrentDifference']].copy()
        export_csv_for_id(meta_df_final_features, "preproc_final")

    print("PreprocSplit function finished")