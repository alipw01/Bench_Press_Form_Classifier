from preprocessing import *
from feature_extraction import *
from classification import *
from rep_counting import *
from collections import Counter
import pickle


def max_consecutive(lst, elem):
    """
    Function for calculating the greatest number of consecutive repetitions of class elem
    param lst: the list of form classifications
    param elem: the class which is searched for within the list
    return: maximum number of consecutive instances of elem within lst
    """
    max_count = 0
    count = 0
    for item in lst:
        if item == elem:
            count += 1
            if count > max_count:
                max_count = count
        else:
            count = 0
    return max_count


def run_retrospective(mimu22bl_file, mimu4844_file):
    """
    Function for running the retrospective version of the model, where information regarding repetitions and bench press form are given retrospectively - given the input of data files
    """
    # Loads the inputted data
    mimu22bl_df = pd.read_csv(mimu22bl_file)
    mimu4844_df = pd.read_csv(mimu4844_file)

    # Loads the trained model
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)

    # Define the window size
    window_size = 2
    # Used for containing repetition and classification values
    reps = 0
    history = []
    # Dictionaries containing class labels and associated corrections
    form_dict = {1: 'Correct Form - Well Done!', 2: 'Incorrect Form - Bar path is angled towards the waist', 3: 'Incorrect Form - Bar path is angled towards the head', 4: 'Incorrect Form - Uneven extension across arms (right arm extending quicker than left arm)', 5: 'Incorrect Form - Uneven extension across arms (left arm extending quicker than right arm)'}
    corrections_dict = {1: 'There are no improvements to be made', 2: 'adjust the angle of extension to be straight up from the chest', 3: 'adjust the angle of extension to be straight up from the chest', 4: 'adjust extension so that the bar is kept level throughout the movement', 5: 'adjust extension so that the bar is kept level throughout the movement'}

    # Pre-process data
    preprocessor_l = SensorDataPreprocessor(data=mimu22bl_df)
    mimu22bl_df = preprocessor_l.main()
    preprocessor_r = SensorDataPreprocessor(data=mimu4844_df)
    mimu4844_df = preprocessor_r.main()

    # Splits the DataFrames into list of smaller DataFrames of x seconds, where x is the window_size
    # Convert the 'TimeStamp' column to datetime format
    mimu22bl_df['TimeStamp'] = pd.to_datetime(mimu22bl_df['TimeStamp'], unit='s')
    mimu4844_df['TimeStamp'] = pd.to_datetime(mimu4844_df['TimeStamp'], unit='s')
    # Set the time interval for splitting the DataFrame
    interval = pd.Timedelta(seconds=window_size)
    # Group the DataFrame by time intervals
    mimu22bl_groups = mimu22bl_df.groupby(pd.Grouper(key='TimeStamp', freq=interval))
    mimu4844_groups = mimu4844_df.groupby(pd.Grouper(key='TimeStamp', freq=interval))
    # Convert each group to a DataFrame and append it to a list
    mimu22bl_df_list = [group[1] for group in mimu22bl_groups]
    mimu4844_df_list = [group[1] for group in mimu4844_groups]
    
    # Checks that the files contain the same number of windows
    if len(mimu22bl_df_list)==len(mimu4844_df_list):
        # Iterates through list of DataFrames
        for i in range(len(mimu22bl_df_list)):
            # Sets the MIMU22BL and MIMU4844 DataFrames from list
            mimu22bl_df = mimu22bl_df_list[i]
            mimu4844_df = mimu4844_df_list[i]
            # Calculates the number of repetitions within the DataFrames and appends it to reps
            counter = Rep_Counter(df1=mimu22bl_df, df2=mimu4844_df, sample_rate=250)
            num_of_peaks = counter.rep_counter()
            reps += num_of_peaks
            # Extracts features from DataFrames
            sensor_data_features = SensorDataFeatures(left=mimu22bl_df, right=mimu4844_df)
            features_df = sensor_data_features.extract_features(window_size)
            # Predicts class
            predicted_class = trained_model.predict(features_df)
            # For each repetition within the data, the form class is appended to history
            for i in range(num_of_peaks):
                history.append(int(predicted_class[0]))
        # Overall statistics regarding the classifications and repetitions are printed
        print('-----------------------------------------------------')
        print('Thank you for using this bench press form classifier. Your final statistics are as follows:')
        print('Total Number of Repetitions: ' + str(reps))
        if len(history)>0:
            counter = Counter(history)
            most_common = form_dict[int(counter.most_common(1)[0][0])]
            correction = corrections_dict[int(counter.most_common(1)[0][0])]
            correct_form_reps = str(history.count(1))
            consecutive_correct_reps = str(max_consecutive(history, 1))
        else:
            most_common = 'N/A - No repetitions were performed'
            correction = 'N/A - No repetitions were performed'
            correct_form_reps = '0'
            consecutive_correct_reps = '0'
        print('Most common form classification: ' + most_common)
        print('Recommended correction based on most common form classification: ' + correction)
        print("Number of repetitions performed with correct form: " + correct_form_reps)
        print("Greatest number of consecutive repetitions performed with correct form: " + consecutive_correct_reps)
        print('-----------------------------------------------------')
    else:
        print('Error: files contain inconsistent timestamps')


if __name__ == '__main__':
    # To run, please input the file path of the associated MIMU22BL and MIMU4844 data files
    run_retrospective(mimu22bl_file='data/cleaned_data/1_1_mimu22bl.csv', mimu4844_file='data/cleaned_data/1_1_mimu4844.csv')