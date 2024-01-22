from load_dans import all_data as data
import pandas as pd

print("Merging data...")
merged_data = []
for i in enumerate(data):
    for j in enumerate(i[1]):
        merged_data.append(j[1])

columns = ['Filename', 'Timestamp', 'Status', 'Phase1Effect', 'Phase2Effect', 'Phase3Effect',
           'Phase1Voltage', 'Phase2Voltage', 'Phase3Voltage', 'Value7', 'ChargingStatus', 'ID', 'Value10']

print("Creating dataframe...")
df = pd.DataFrame(merged_data, columns=columns)
# Sort by ID
df = df.sort_values(by=['ID'])
# Reset index
df = df.reset_index(drop=True)

# Specify the desired ID
desired_id = "0"  # Replace this with the ID you want

# Filter DataFrame based on the desired ID
desired_rows = df[df['ID'] == desired_id]

# Print the filtered DataFrame
print(desired_rows)
