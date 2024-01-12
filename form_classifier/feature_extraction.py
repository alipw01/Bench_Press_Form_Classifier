import pandas as pd
import numpy as np


class SensorDataFeatures:
    def __init__(self, left, right, label=None):
        """
        param left: the mimu22bl DataFrame
        param right: the mimu4844 DataFrame
        param label: optional parameter containing the labels for the features
        """
        self.left = left
        self.right = right
        self.label = label
    
    def extract_features(self, window_size=2):
        """
        Function for applying the feature extraction process
        param window_size: fixed-length window size for extracting the features
        return: the DataFrame containing the extracted features
        """
        # Convert timestamps to datetime objects
        self.left.loc[:, 'TimeStamp'] = pd.to_datetime(self.left['TimeStamp'], unit='s')
        self.right.loc[:, 'TimeStamp'] = pd.to_datetime(self.right['TimeStamp'], unit='s')
        
        # Determine window start and end times
        left_start_time = self.left['TimeStamp'].min()
        left_end_time = self.left['TimeStamp'].max()
        left_window_start_times = pd.date_range(start=left_start_time, end=left_end_time, freq=pd.Timedelta(seconds=window_size))
        left_window_end_times = left_window_start_times + pd.Timedelta(seconds=window_size)
        right_start_time = self.right['TimeStamp'].min()
        right_end_time = self.right['TimeStamp'].max()
        right_window_start_times = pd.date_range(start=right_start_time, end=right_end_time, freq=pd.Timedelta(seconds=window_size))
        right_window_end_times = right_window_start_times + pd.Timedelta(seconds=window_size)
        
        features = []
        # Get windows from both sensors using the 'TimeStamp' column
        for left_start, left_end, right_start, right_end in zip(left_window_start_times, left_window_end_times, right_window_start_times, right_window_end_times):
            left_window = self.left[(self.left['TimeStamp'] >= left_start) & (self.left['TimeStamp'] < left_end)]
            right_window = self.right[(self.right['TimeStamp'] >= right_start) & (self.right['TimeStamp'] < right_end)]
            
            if not (left_window.empty or right_window.empty):
                    # Checks that the window is approximately the specified length
                    left_features = self.extract_window_features(left_window)
                    right_features = self.extract_window_features(right_window)
                    # Appends features to the features list - along with the label if one was provided
                    if self.label is not None:
                        features.append([self.label] + left_features + right_features)
                    else:
                        features.append(left_features + right_features)
        
        # Combine features into a new pandas dataframe
        feature_names = ['mean', 'std', 'rms']
        column_names = []
        if self.label is not None:
            column_names.append('label')
        for col in self.left.columns:
            if col not in ['Packet No.', 'TimeStamp']:
                for f in feature_names:
                    column_names.append('left_' + col + '_' + f)       
        for col in self.right.columns:
            if col not in ['Packet No.', 'TimeStamp']:
                for f in feature_names:
                    column_names.append('right_' + col + '_' + f)
        df = pd.DataFrame(features, columns=column_names)
        
        return df
    
    def extract_window_features(self, window):
        """
        Function for calculating each feature from the given window
        param window: DataFrame of x seconds worth of data, where x is the window_size in extract_features
        return: list of extracted features
        """
        # Calculates features for each column within the DataFrame
        features = []
        for col in window.columns:
            if col not in ['Packet No.', 'TimeStamp']:
                values = window[col].values
                # Calculates mean
                mean = np.mean(values)
                # Calculates standard deviation
                std = np.std(values)
                # Calculates root mean squared
                rms = np.sqrt(np.mean(np.square(values)))
                # Appends features to features list
                features += [mean, std, rms]
        return features