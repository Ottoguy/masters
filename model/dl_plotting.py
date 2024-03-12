import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def DeepLearningPlotting():
    # Specify the directory where your files are located
    folder_path = 'prints/deep_learning/'
    # Create a pattern to match files in the specified format
    file_pattern = '*'
    # Get a list of all files matching the pattern
    file_list = glob.glob(os.path.join(folder_path, file_pattern))
    # Sort the files based on modification time (latest first)
    file_list.sort(key=os.path.getmtime, reverse=True)
    # Take the latest file
    latest_file = file_list[0]
    # Load your data from the latest file
    #df = pd.read_csv(latest_file)
    df = pd.read_csv('prints/deep_learning/ts_samples_60_clusters_10_test_size_0.3_epochs_400_batch_size_32_layer1_units_128_layer2_units_64_dropout_rate_0.4_20240312_124401.csv')

    #Drop the Barebones and immediate columns
    df = df.drop(['Barebones', 'Immediate'], axis=1)

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Plot each column as a line
    for column in df.columns:
        if column == 'Real':
            plt.plot(df[column], linewidth=2.5, label=column)
        else:
            plt.plot(df[column], label=column)

    # Add labels and title
    plt.xlabel('Row Number')
    plt.ylabel('Value')
    plt.title('Deep Learning Plot')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Call the function
DeepLearningPlotting()
