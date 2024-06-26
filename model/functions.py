import numpy as np
from datetime import datetime
import os

# Encode hours and minutes as a combined cyclical feature
def encode_cyclical_features(df, column_name):
    df[column_name + '_sin'] = np.sin(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))
    df[column_name + '_cos'] = np.cos(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))

def export_csv_for_id(df, id_to_export, ts_samples=0):
    parent_folder="prints"
    id_prints = parent_folder + "/id"
    # If the desired ID is "all", export all rows
    if id_to_export.lower() == "all":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "all")
    # If the desired ID is "meta", export the meta_df
    elif id_to_export.lower() == "meta":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "meta")
    elif id_to_export.lower() == "extracted":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "extracted", str(ts_samples))
    elif id_to_export.lower() == "filtered":
        desired_rows = df.copy()
    elif id_to_export.lower() == "preproc_immediate":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "preproc_immediate")
    elif id_to_export.lower() == "preproc_intermediate":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "preproc_intermediate")
    elif id_to_export.lower() == "preproc_final":
        desired_rows = df.copy()
        output_folder = os.path.join(parent_folder, "preproc_final")
    else:
        # Filter DataFrame based on the desired ID
        desired_rows = df[df['ID'] == id_to_export]
        output_folder = os.path.join(id_prints, str(id_to_export))

    # Create a folder named "prints" if it doesn't exist
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)

    # Create a subfolder with the ID or "all" as its name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the current date and time
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the file name
    output_file = f"{output_folder}/{current_datetime}.csv"

    # Print desired_rows to a CSV file
    desired_rows.to_csv(output_file, index=False)

    if id_to_export.lower() == "meta":
        print(f"Meta exported to: {output_file}")
    elif id_to_export.lower() == "all":
        print(f"All rows exported to: {output_file}")
    else:
        print(f"Exported rows for ID {id_to_export} to: {output_file}")