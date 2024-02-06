import numpy as np

# Encode hours and minutes as a combined cyclical feature
def encode_cyclical_features(df, column_name):
    df[column_name + '_sin'] = np.sin(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))
    df[column_name + '_cos'] = np.cos(2 * np.pi * (df[column_name].dt.hour * 60 + df[column_name].dt.minute) / (24 * 60))
    