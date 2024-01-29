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

# Group the data by day of the week and hour of the day, then calculate the average occupancy
average_occupancy_by_time = df.groupby(['DayOfWeek', 'HourOfDay']).size().unstack().mean()

# Plotting the data
fig, ax = plt.subplots(figsize=(15, 6))

# Bar plot for average occupancy by time
average_occupancy_by_time.plot(kind='bar', color='skyblue', ax=ax)
ax.set_title('Average Occupancy by Time (Across All Days)')
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Average Number of EV Occupancies')
ax.set_xticklabels(range(24), rotation=0)
ax.legend(loc='upper right')

plt.show()
