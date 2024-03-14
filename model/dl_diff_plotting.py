import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def DeepLearningDiffPlotting():
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
    df = pd.read_csv(latest_file)

    # Calculate absolute differences between each column and the "Real" column
    for column in df.columns:
        if column != 'Real':
            df[column + '_abs_diff'] = abs(float(df['Real']) - float(df[column]))

    # Plotting
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    # Plot the absolute differences
    for column in df.columns:
        if '_abs_diff' in column:
            plt.plot(df[column], label=column)

    # Add labels and title
    plt.xlabel('Row Number')
    plt.ylabel('Absolute Difference from Real')
    plt.title('Deep Learning Plot - Absolute Differences from Real')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

# Call the function
DeepLearningDiffPlotting()
