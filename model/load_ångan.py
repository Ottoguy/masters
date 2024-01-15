import pandas as pd
import glob

# Specify the path to your folder containing CSV files
folder_path = 'data/Ångström/Charger_Data/spec/*.csv'

# Get a list of file names matching the pattern
csv_files = glob.glob(folder_path)

# Initialize an empty list to store DataFrames
dfs = []

# Iterate through the list of CSV files
for csv_file in csv_files:
    print(f"Loading data from file: {csv_file}")

    # Read the CSV file into a Pandas DataFrame
    data = pd.read_csv(csv_file)

    # Modify column names (replace with your logic)
    data.columns = [f"{csv_file}_{col}" for col in data.columns]

    # Append the DataFrame to the list
    dfs.append(data)

print("Data load completed for all CSV files in the folder.")

# Concatenate the list of DataFrames into a single DataFrame vertically
merged_data = pd.concat(dfs, axis=1, ignore_index=True)

# Ensure each CSV contributes up to 12 columns, with extra columns filled with zeros
columns_to_add = 12 - merged_data.shape[1]
if columns_to_add > 0:
    for i in range(columns_to_add):
        merged_data[f"extra_{i+1}"] = 0

# Reshape the DataFrame by creating a new line after every 12 columns
reshaped_data = merged_data.groupby(
    merged_data.columns // 12, axis=1).apply(lambda x: pd.concat([col for col in x], axis=1))

# Display information about the reshaped DataFrame
print("\nReshaped Data:")
print("Data type: ", type(reshaped_data))
print("Data shape: ", reshaped_data.shape)
print("Data columns: ", reshaped_data.columns)
# Display the first few rows of the reshaped DataFrame
print(reshaped_data.head())
