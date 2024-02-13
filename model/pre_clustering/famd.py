import pandas as pd
import os
import glob
from prince import FAMD
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

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

# Drop unnecessary columns
columns_to_drop = ['ID', 'TimeConnected', 'TimeDisconnected']
data = data.drop(columns=columns_to_drop)

cumulative_variance_list = []
# Iterate over different values of n_components
for n_components in range(2, 15):
    explained_variance = None
    cumulative_variance = None
    # Perform Factor Analysis of Mixed Data (FAMD)
    famd = FAMD(
        n_components=n_components,
        n_iter=3,
        copy=True,
        check_input=True,
        random_state=42,
        engine="sklearn",
        handle_unknown="error"
    )
    famd = famd.fit(data)

    # Calculate explained variance manually
    explained_variance = famd.eigenvalues_ / famd.eigenvalues_.sum()
    cumulative_variance = explained_variance.cumsum()
    cumulative_variance_list.append(cumulative_variance[-2])

print(cumulative_variance_list)
# Plot cumulative variance explained
plt.plot(range(1, len(cumulative_variance_list) + 1), cumulative_variance_list, label=f'n_components = {n_components}')

# Add labels and title
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained for Different n_components')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Display the principal coordinates
print("Principal Coordinates:\n", famd.row_coordinates(data))
print(famd.eigenvalues_summary)
print(famd.column_coordinates_)
print(famd.row_contributions_.sort_values(0, ascending=False).head(5).style.format('{:.3%}'))
print(famd.column_contributions_.style.format('{:.0%}'))