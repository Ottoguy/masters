import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

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
df = pd.read_csv(latest_file)

# Convert the 'TimeConnected' and 'TimeDisconnected' columns to datetime objects
df['TimeConnected'] = pd.to_datetime(df['TimeConnected'])
df['TimeDisconnected'] = pd.to_datetime(df['TimeDisconnected'])

# Create a new column for the day of the week
df['DayOfWeek'] = df['TimeConnected'].dt.day_name()

# Order the days of the week chronologically
days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['DayOfWeek'] = pd.Categorical(df['DayOfWeek'], categories=days_order, ordered=True)

# Create a new column for the hour of the day
df['HourOfDay'] = df['TimeConnected'].dt.hour

# Create a DataFrame to represent occupancy by minute
occupancy_by_minute = pd.DataFrame(index=pd.date_range(start='00:00', end='23:59', freq='T'))

# Populate the DataFrame with occupancy data
for index, row in df.iterrows():
    start_time = row['TimeConnected']
    end_time = row['TimeDisconnected']
    occupancy_by_minute.loc[start_time:end_time, index] = 1

# Group by day of the week and hour of the day, then calculate the sum to get total occupancy
total_occupancy_by_time = occupancy_by_minute.groupby([occupancy_by_minute.index.day_name(), occupancy_by_minute.index.hour]).sum()

# Plotting the data
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), sharex=True, sharey=True)

# Flatten the 2x4 array of subplots into a 1D array for easier indexing
axes = axes.flatten()

for i, day in enumerate(days_order):
    total_occupancy_by_time.loc[day].plot(kind='bar', color='skyblue', ax=axes[i])
    axes[i].set_title(day)
    axes[i].set_xlabel('Hour of the Day')
    axes[i].set_ylabel('Number of EV Occupancies')

# Additional subplot for the average across all days
average_across_all_days = total_occupancy_by_time.mean(level=total_occupancy_by_time.index.names.index('hour'))
axes[-1].bar(average_across_all_days.index, average_across_all_days[0], color='skyblue')
axes[-1].set_title('Average Across All Days')
axes[-1].set_xlabel('Hour of the Day')
axes[-1].set_ylabel('Number of EV Occupancies')

# Adjust layout for better readability
plt.tight_layout()
plt.show()
