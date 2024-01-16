import os
import csv

# Specify the path to your folder containing CSV files
folder_path = 'data/Dansmästaren/Charger_Data/'

# Get a list of all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Initialize a dictionary to store data grouped by filename
data_by_filename = {}

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

        # Group data by filename in the dictionary
        if csv_file not in data_by_filename:
            data_by_filename[csv_file] = []
        data_by_filename[csv_file].extend(file_data_with_filename)

# Convert the dictionary values to a list
all_data = list(data_by_filename.values())

# Now 'all_data' contains one list for every filename, each containing lists of data
for i in range(100):
    print(all_data[1][i])
# Ok du märker att vid denna utskrift byts bil efter ett par rader, känns bra att visualisera de 6 värdena för att påvisa det? Sen så ser du i slutet att värdena fastnar om den går offline, värt att ha i åtanke
