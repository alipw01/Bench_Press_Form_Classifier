import pickle
from tqdm import tqdm
from preprocessing import *
from data_splitting import *
from classification import *
from classifier_evaluation import *
from itertools import combinations
import pandas as pd
from scipy.stats import skew, kurtosis


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
                left_window_first = left_window.loc[left_window.index[0], 'TimeStamp'].timestamp()
                left_window_last = left_window.loc[left_window.index[-1], 'TimeStamp'].timestamp()
                right_window_first = right_window.loc[right_window.index[0], 'TimeStamp'].timestamp()
                right_window_last = right_window.loc[right_window.index[-1], 'TimeStamp'].timestamp()
                if round((left_window_last-left_window_first), 1) == window_size and round((right_window_last-right_window_first), 1) == window_size:
                    # Extracts the features
                    left_features = self.extract_window_features(left_window)
                    right_features = self.extract_window_features(right_window)
                    # Appends features to the features list - along with the label if one was provided
                    if self.label is not None:
                        features.append([self.label] + left_features + right_features)
                    else:
                        features.append(left_features + right_features)
        
        # Combine features into a new pandas dataframe
        feature_names = ['mean', 'std', 'iqr', 'mad', 'rms', 'skew', 'kurt']
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
                # Calculates inter-quartile range
                iqr = np.subtract(*np.percentile(values, [75, 25]))
                # Calculates mean absolute deviation
                mad = np.mean(np.abs(np.subtract(values, np.mean(values))))
                # Calculates root mean squared
                rms = np.sqrt(np.mean(np.square(values)))
                # Calculates skewness
                skewness = skew(values)
                # Calculates kurtosis
                kurt = kurtosis(values)
                # Appends features to features list
                features += [mean, std, iqr, mad, rms, skewness, kurt]
        return features


if __name__ == '__main__':
    # Define the list of possible features
    features = ['mean', 'std', 'iqr', 'mad', 'rms', 'skew', 'kurt']

    # Create a list of all possible combinations of features to try
    feature_combinations = []
    for i in range(1, len(features) + 1):
        feature_combinations.extend(list(combinations(features, i)))

    # Initialize the variables to keep track of the best set of features and its accuracy score
    best_features = []
    best_evaluations = []
    best_accuracy = 0

    # Define window size
    window_size = 2

    # Split data
    data_splitter = DataSplitter('./data/cleaned_data/')
    training, testing = data_splitter.split_data_rolling_window()

    training_features = []
    testing_features = []
    # Pre-processes and extracts features for training set
    for pair in training:
        # Checks that the DataFrames are the expected DataFrames and that their form ID and participant ID match 
        mimu22bl_df = pair[0]
        mimu4844_df = pair[1]
        mimu22bl_participant_id = mimu22bl_df.values[0][-1]
        mimu22bl_form_id = mimu22bl_df.values[0][-2]
        mimu4844_participant_id = mimu4844_df.values[0][-1]
        mimu4844_form_id = mimu4844_df.values[0][-2]
        if mimu22bl_participant_id == mimu4844_participant_id and mimu22bl_form_id == mimu4844_form_id:
            form_id = mimu22bl_form_id
        else:
            print('Error: Sensor Data Does Not Match Up')
            pass
        # Drops irrelevant columns
        mimu22bl_df = mimu22bl_df.drop('Form_ID', axis=1)
        mimu22bl_df = mimu22bl_df.drop('Participant_ID', axis=1)
        mimu4844_df = mimu4844_df.drop('Form_ID', axis=1)
        mimu4844_df = mimu4844_df.drop('Participant_ID', axis=1)
        # Preprocess data
        preprocessor_l = SensorDataPreprocessor(data=mimu22bl_df)
        mimu22bl_df = preprocessor_l.main()
        preprocessor_r = SensorDataPreprocessor(data=mimu4844_df)
        mimu4844_df = preprocessor_r.main()
        # Extract the features from the mimu22bl and mimu4844 DataFrames
        sensor_data_features = SensorDataFeatures(left=mimu22bl_df, right=mimu4844_df, label=form_id)
        features_df = sensor_data_features.extract_features(window_size)
        # Append the feature dataframe to the list of feature DataFrames
        training_features.append(features_df)
    
    # Pre-processes and extracts features for testing set
    for pair in testing:
        # Checks that the DataFrames are the expected DataFrames and that their form ID and participant ID match 
        mimu22bl_df = pair[0]
        mimu4844_df = pair[1]
        mimu22bl_participant_id = mimu22bl_df.values[0][-1]
        mimu22bl_form_id = mimu22bl_df.values[0][-2]
        mimu4844_participant_id = mimu4844_df.values[0][-1]
        mimu4844_form_id = mimu4844_df.values[0][-2]
        if mimu22bl_participant_id == mimu4844_participant_id and mimu22bl_form_id == mimu4844_form_id:
            form_id = mimu22bl_form_id
        else:
            print('Error: Sensor Data Does Not Match Up')
            pass
        # Drops irrelevant columns
        mimu22bl_df = mimu22bl_df.drop('Form_ID', axis=1)
        mimu22bl_df = mimu22bl_df.drop('Participant_ID', axis=1)
        mimu4844_df = mimu4844_df.drop('Form_ID', axis=1)
        mimu4844_df = mimu4844_df.drop('Participant_ID', axis=1)
        # Preprocess data
        preprocessor_l = SensorDataPreprocessor(data=mimu22bl_df)
        mimu22bl_df = preprocessor_l.main()
        preprocessor_r = SensorDataPreprocessor(data=mimu4844_df)
        mimu4844_df = preprocessor_r.main()
        # Extract the features from the mimu22bl and mimu4844 DataFrames
        sensor_data_features = SensorDataFeatures(left=mimu22bl_df, right=mimu4844_df, label=form_id)
        features_df = sensor_data_features.extract_features(window_size)
        # Append the feature dataframe to the list of feature DataFrames
        testing_features.append(features_df)
    
    # Creates training and testing DataFrames containing extracted features
    training = pd.concat(training_features)
    testing = pd.concat(testing_features)

    # Iterate through all possible combinations of features
    for feature_set in tqdm(feature_combinations):
        train = training
        test = testing
        # Remove the columns that have the features that are not included in the current set
        columns_to_remove = [col for col in train.columns if col.endswith(tuple(set(features) - set(feature_set)))]
        features_to_remove = list(set(train.columns) & set(columns_to_remove + ['label']))
        x_train = train.drop(features_to_remove, axis=1)
        y_train = train['label']
        # Train model and calculate mean cross-validated accuracy
        model = ClassifierCrossValidation()
        cross_val_mean = model.train(x_train, y_train)
        # Update the best set of features and its accuracy score if the current set is better
        if cross_val_mean > best_accuracy:
            best_features = []
            best_features.append(feature_set)
            best_accuracy = cross_val_mean
        elif cross_val_mean == best_accuracy:
            best_features.append(feature_set)
            best_accuracy = cross_val_mean

    # Print the best set of features and its accuracy score
    print('Best features:', best_accuracy, best_features)
    # Train classifier using features achieving highest mean cross-validated accuracy and evaluate its performance on testing set
    for features in best_features:
        columns_to_remove = [col for col in train.columns if col.endswith(tuple(set(features) - set(feature_set)))]
        features_to_remove = list(set(train.columns) & set(columns_to_remove + ['label']))
        x_train = training.drop(features_to_remove, axis=1)
        y_train = training['label']
        x_test = testing.drop(features_to_remove, axis=1)
        y_test = testing['label']
        model = ClassifierCrossValidation()
        cross_val_mean = model.train(x_train, y_train)
        y_pred = model.predict(x_test)
        evaluator = ModelEvaluator(y_pred, y_test)
        accuracy, precision, recall, f1, cm = evaluator.evaluate()