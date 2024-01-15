import os
import csv

# Specify the path to your folder containing CSV files
folder_path = 'data/Dansmästaren/Charger_Data/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize a list to store data from all CSV files
all_data = []

# Loop through each CSV file and read its content
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)

    with open(file_path, 'r', newline='') as file:
        csv_reader = csv.reader(file)

        # Skip the header if needed
        # next(csv_reader)

        # Read the data and append it to the list
        file_data = list(csv_reader)
        all_data.extend(file_data)

# Now 'all_data' contains the data from all CSV files
for i in range(10):
    print(all_data[i])
