import matplotlib.pyplot as plt
import csv
from datetime import datetime

# from load_xlsx import data as xlsx_data
# from load_ångan import data as ångan_data
from load_dans import all_data as dans_data

# Choose which data to ue
data = dans_data
print(len(data))

# Remove the first row
data = data[1:]
doc1 = data[0]
print(doc1[0])

# Extract relevant data for plotting
timestamps = []
values_4 = []
values_5 = []
values_6 = []

for row in doc1:
    time_str = row[1][11:]  # Extract time without the first 11 characters
    timestamps.append(time_str)
    values_4.append(float(row[3]))
    values_5.append(float(row[4]))
    values_6.append(float(row[5]))

# Plot the data
plt.plot(timestamps, values_4, label='Element 4')
plt.plot(timestamps, values_5, label='Element 5')
plt.plot(timestamps, values_6, label='Element 6')

# Set the title of the graph
plt.title(doc1[0][0])

# Add labels and legend
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()

# Display every 10th timestamp on the x-axis
plt.xticks(timestamps[::20])

# Show the plot
plt.show()
