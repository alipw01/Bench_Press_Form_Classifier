import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from preprocessing import *


class Rep_Counter:
    def __init__(self, df1, df2, sample_rate):
        """
        param df1: DataFrame of the left (mimu22bl) sensor
        param df2: DataFrame of the right (mimu4844) sensor
        param sample_rate: the sample rate which each DataFrame is re-sampled to
        """
        self.df1 = df1
        self.df2 = df2
        self.sample_rate = sample_rate


    def rep_counter(self):
        """
        Function for calculating the number of repetitions within a DataFrame
        return: number of peaks
        """
        self.df1.loc[:, 'TimeStamp'] = pd.to_datetime(self.df1.loc[:, 'TimeStamp'], unit='s')
        self.df2.loc[:, 'TimeStamp'] = pd.to_datetime(self.df2.loc[:, 'TimeStamp'], unit='s')

        # Resample the dataframes
        self.df1 = self.df1.set_index('TimeStamp').resample('{}ms'.format(self.sample_rate)).mean().reset_index()
        self.df2 = self.df2.set_index('TimeStamp').resample('{}ms'.format(self.sample_rate)).mean().reset_index()

        # Concatenate the two dataframes on TimeStamp column
        combined_df = pd.merge(self.df1, self.df2, on='TimeStamp')

        # Select only the columns with sensor data
        sensor_cols = ['a00 m/s^2_x', 'a10 m/s^2_x', 'a20 m/s^2_x', 'g00 deg/s_x', 'g10 deg/s_x', 'g20 deg/s_x',
                    'a00 m/s^2_y', 'a10 m/s^2_y', 'a20 m/s^2_y', 'g00 deg/s_y', 'g10 deg/s_y', 'g20 deg/s_y']
        sensor_df = combined_df[sensor_cols]

        # Apply PCA
        pca = PCA(n_components=1)
        pca_component = pca.fit_transform(sensor_df)

        # Add PCA component to the resampled dataframe
        combined_df['PCA_Component'] = pca_component
        df = combined_df[['TimeStamp', 'PCA_Component']]
        df = df.set_index('TimeStamp')

        # Calculate the peaks within the DataFrame
        peaks, _ = find_peaks(df['PCA_Component'], height=25, distance=5)

        return len(peaks)