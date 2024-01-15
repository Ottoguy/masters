import os
import csv

# Specify the path to your folder containing CSV files
folder_path = 'data/Dansmästaren/Charger_Data/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize a list to store data from all CSV files
all_data = [["filename", "timestamp", "availability",
            "a_fas1", "a_fas2", "a_fas3", "b_fas1", "b_fas2", "b_fas3", "unknown", "empty", "connection", "id"]]

# Loop through each CSV file and read its content
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)

        # Read the data
        file_data = list(csv_reader)

        # Add the filename as the first column in each row
        file_data_with_filename = [[csv_file] + row for row in file_data]

        # Check and fill missing values with "None"
        for row in file_data_with_filename:
            while len(row) < 13:
                row.append(None)

        # Append the data to the list
        all_data.extend(file_data_with_filename)

# Now 'all_data' contains the data from all CSV files with the filename as the first column
for i in range(10):
    print(all_data[i])
