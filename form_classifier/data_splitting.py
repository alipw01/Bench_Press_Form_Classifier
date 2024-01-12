import pandas as pd
import os

class DataSplitter:
    def __init__(self, path, split_ratio=0.9):
        """
        param path: the path of the directory containing the data files
        param split_ratio: the percentage of each file appended to the training set
        """
        self.path = path
        self.split_ratio = split_ratio

    def split_data_rolling_window(self):
        """
        Function for splitting the data within a folder into a training set and testing set
        return: training_set and testing_set lists containing associated mimu22bl and mimu4844 DataFrames
        """
        # Creates lists for containing training and testing data 
        train = []
        test = []
        # Creates lists for containing the associated mimu22bl and mimu4844 DataFrames in individual lists
        training_set = []
        testing_set = []
        # Loop through all files in the directory
        for filename in os.listdir(self.path):
            # Load file and set form ID and participant ID using the filename
            file = pd.read_csv(self.path + filename)
            file['Form_ID'] = filename.split('_')[1]
            file['Participant_ID'] = filename.split('_')[0]
            # Splits first 90% of file into train and last 10% into test
            rows = len(file)
            train_rows = int(rows * self.split_ratio)
            train.append(file.iloc[:train_rows])
            test.append(file.iloc[train_rows:])
        # Appends each associated mimu22bl and mimu4844 within the train or test sets to the training_set or testing_set lists, respectively
        for i in range(0, len(train), 2):
            training_set.append(train[i:i+2])
        for i in range(0, len(test), 2):
            testing_set.append(test[i:i+2])
        return training_set, testing_set