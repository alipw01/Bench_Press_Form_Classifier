import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

class SensorDataPreprocessor:
    def __init__(self, data):
        self.data = data


    def remove_outliers(self):
        """
        Function for removing any data which isn't within three standard deviations from the mean of each accelerometer or gyroscope axis
        return: self.data pandas DataFrame
        """
        self.data = self.data[(np.abs(self.data["a00 m/s^2"] - self.data["a00 m/s^2"].mean()) <= (3 * self.data["a00 m/s^2"].std())) &
                            (np.abs(self.data["a10 m/s^2"] - self.data["a10 m/s^2"].mean()) <= (3 * self.data["a10 m/s^2"].std())) &
                            (np.abs(self.data["a20 m/s^2"] - self.data["a20 m/s^2"].mean()) <= (3 * self.data["a20 m/s^2"].std())) &
                            (np.abs(self.data["g00 deg/s"] - self.data["g00 deg/s"].mean()) <= (3 * self.data["g00 deg/s"].std())) &
                            (np.abs(self.data["g10 deg/s"] - self.data["g10 deg/s"].mean()) <= (3 * self.data["g10 deg/s"].std())) &
                            (np.abs(self.data["g20 deg/s"] - self.data["g20 deg/s"].mean()) <= (3 * self.data["g20 deg/s"].std()))]
        return self.data


    def remove_null_values(self):
        """
        Function for removing any null values within the pandas DataFrame
        return: self.data pandas DataFrame
        """
        self.data.dropna(inplace=True)
        return self.data
    

    def apply_savitzky_golay_filter(self, df, window_size, order, deriv=0, rate=1):
        """
        Applies the Savitzky-Golay filter on the pandas DataFrame.
        param df: pandas Dataframe
        param window_size: the number of adjacent data points to use for fitting the polynomial
        param order: the degree of the polynomial to fit
        param deriv: the order of the derivative to fit
        param rate: sampling rate of the input data
        return: pandas DataFrame
        """
        filtered_data = np.apply_along_axis(savgol_filter, axis=0, arr=df.values, window_length=window_size, polyorder=order, deriv=deriv, delta=rate)
        return pd.DataFrame(filtered_data, columns=df.columns)
    

    def main(self):
        """
        Function for running the pre-processing stage
        return: self.data pandas DataFrame
        """
        self.remove_null_values()
        self.apply_savitzky_golay_filter(self.data, 21, 4, 0, 1)
        self.remove_outliers()
        return self.data