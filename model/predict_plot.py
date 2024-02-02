import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import mplcursors  # Import the mplcursors library

# Specify the directory where your files are located
folder_path = 'prints/meta/'

# Create a pattern to match files in the specified format
file_pattern = '*'

# Get a list of all files matching the pattern
file_list = glob.glob(os.path.join(folder_path, file_pattern))

# Sort the files based on modification time (latest first)
file_list.sort(key=os.path.getmtime, reverse=True)

# Take the latest file
latest_file = file_list[0]

# Load your data from the latest file
data = pd.read_csv(latest_file)

print("CSV file loaded")

# Columns to exclude
exclude_columns = ['ID', 'Half_Minutes', 'Current_Type', 'FullyCharged', 'Weekend']

# Columns to plot against Half_Minutes
x_columns = [col for col in data.columns if col not in exclude_columns]

# Calculate the number of subplots needed
num_subplots = len(x_columns)
num_rows = num_subplots // 2 + num_subplots % 2  # Use 2 columns for each row

# Create subplots
fig, axes = plt.subplots(num_rows, 2, figsize=(15, 5 * num_rows))

# Flatten the 2D array of subplots
axes = axes.flatten()

# Differentiate 'FullyCharged' and 'Current_Type' with colors
colors = data.apply(lambda row: 'green' if (row['FullyCharged'] and row['Current_Type'] == '3-Phase') else
                                    'blue' if (row['FullyCharged'] and row['Current_Type'] == '1-Phase') else
                                    'red' if (not row['FullyCharged'] and row['Current_Type'] == '3-Phase') else
                                    'orange' if (not row['FullyCharged'] and row['Current_Type'] == '1-Phase') else 'black', axis=1)

# Plotting each column against Half_Minutes with a logarithmic y-axis
for i, x_column in enumerate(x_columns):
    row = i // 2
    col = i % 2
    
    # Separate data for weekend and non-weekend points
    weekend_data = data[data['Weekend']]
    non_weekend_data = data[~data['Weekend']]
    
    # Plot non-weekend points with circles
    scatter1 = axes[i].scatter(non_weekend_data[x_column], non_weekend_data['Half_Minutes'], c=colors[~data['Weekend']], marker='o', label='FullyCharged and Current_Type (Weekday)')
    
    # Plot weekend points with squares
    scatter2 = axes[i].scatter(weekend_data[x_column], weekend_data['Half_Minutes'], c=colors[data['Weekend']], marker='s', label='FullyCharged and Current_Type (Weekend)')
    
    axes[i].set_title(f'Scatter Plot: Half_Minutes vs {x_column}')
    axes[i].set_xlabel(x_column)
    axes[i].set_ylabel('Half_Minutes')
    axes[i].legend()
    axes[i].set_yscale('log')  # Set y-axis to logarithmic scale
    
    # Enable hover over points to display ID
    mplcursors.cursor(scatter1, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"ID: {non_weekend_data['ID'].iloc[sel.target.index]}"))
    mplcursors.cursor(scatter2, hover=True).connect("add", lambda sel: sel.annotation.set_text(f"ID: {weekend_data['ID'].iloc[sel.target.index]}"))

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()