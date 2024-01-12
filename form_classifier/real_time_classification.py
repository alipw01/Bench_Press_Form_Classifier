from mimu22bl import *
from mimu4844 import *
from preprocessing import *
from feature_extraction import *
from data_splitting import *
from classification import *
from classifier_evaluation import *
from rep_counting import *
import multiprocessing
from collections import Counter
import pickle

# INSTRUCTIONS
# Start - Press *Space Bar*
# Finish - Press 'Q' Twice

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


def classify(queue1, queue2, window_size):
    """
    Function for performing the real-time classification, using the data being optained from the two sensors
    param queue1: queue of recorded data from mimu22bl sensor
    param queue2: queue of recorded data from mimu4844 sensor
    param window_size: the window size for the feature extraction phase
    """
    # Load trained model
    with open('trained_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    # Initialise reps variable and history list for recording the obtained repetitions and classifications
    reps = 0
    history = []
    # Define the columns of the DataFrame
    columns = ['Packet No.', 'TimeStamp', 'a00 m/s^2', 'a10 m/s^2', 'a20 m/s^2', 'g00 deg/s', 'g10 deg/s', 'g20 deg/s']
    # Define the class names and descriptions for the classification
    form_dict = {1: 'Correct Form - Well Done!', 2: 'Incorrect Form - Bar path is angled towards the waist', 3: 'Incorrect Form - Bar path is angled towards the head', 4: 'Incorrect Form - Uneven extension across arms (right arm extending quicker than left arm)', 5: 'Incorrect Form - Uneven extension across arms (left arm extending quicker than right arm)'}
    corrections_dict = {1: 'There are no improvements to be made', 2: 'adjust the angle of extension to be straight up from the chest', 3: 'adjust the angle of extension to be straight up from the chest', 4: 'adjust extension so that the bar is kept level throughout the movement', 5: 'adjust extension so that the bar is kept level throughout the movement'}
    # Create the DataFrames to contain the data for each sensor
    mimu22bl_data = pd.DataFrame(columns=columns)
    mimu4844_data = pd.DataFrame(columns=columns)
    # Set variables used for controlling the collection of data from the queues
    count = 0
    mimu22bl_window_full = False
    mimu4844_window_full = False
    while keyboard.is_pressed("q") == False:
        # Repeat while data is present within the queues
        while not queue1.empty() and not queue2.empty():
            # Iteratively obtain data from the queues until the TimeStamp difference in the data within the DataFrame is greater than or equal to the window size
            if count == 0:
                mimu22bl_packets = queue1.get()[0:8]
                mimu22bl_data = mimu22bl_data.append(pd.Series(mimu22bl_packets, index=mimu22bl_data.columns), ignore_index=True)
                mimu4844_packets = queue2.get()[0:8]
                mimu4844_data = mimu4844_data.append(pd.Series(mimu4844_packets, index=mimu4844_data.columns), ignore_index=True)
            else:
                if mimu22bl_data.iloc[-1]['TimeStamp'] - mimu22bl_data.iloc[0]['TimeStamp'] < window_size:
                    mimu22bl_packets = queue1.get()[0:8]
                    mimu22bl_data = mimu22bl_data.append(pd.Series(mimu22bl_packets, index=mimu22bl_data.columns), ignore_index=True)
                else:
                    mimu22bl_window_full = True
                if mimu4844_data.iloc[-1]['TimeStamp'] - mimu4844_data.iloc[0]['TimeStamp'] < window_size:
                    mimu4844_packets = queue2.get()[0:8]
                    mimu4844_data = mimu4844_data.append(pd.Series(mimu4844_packets, index=mimu4844_data.columns), ignore_index=True)
                else: 
                    mimu4844_window_full = True
            count += 1
            # Once the windows are full, the analysis phase begins
            if mimu22bl_window_full == True and mimu4844_window_full == True:
                # Pre-processes the DataFrame
                preprocessor_l = SensorDataPreprocessor(data=mimu22bl_data)
                mimu22bl_df = preprocessor_l.main()
                preprocessor_r = SensorDataPreprocessor(data=mimu4844_data)
                mimu4844_df = preprocessor_r.main()
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
                # Prints information if more than zero repetitions were performed, else a filler message is printed
                if num_of_peaks > 0:
                    print('-----------------------------------------------------')
                    print('Total Number of Repetitions Detected: ' + str(reps))
                    print('Form Classification: ' + form_dict[int(predicted_class[0])])
                    print('Required Correction: ' + corrections_dict[int(predicted_class[0])])
                    print('-----------------------------------------------------')
                else:
                    print('-----------------------------------------------------')
                    print('Waiting For Repetition To Be Completed...')
                    print('-----------------------------------------------------')
                # Resets values
                mimu22bl_data = pd.DataFrame(columns=columns)
                mimu4844_data = pd.DataFrame(columns=columns)
                count = 0
                mimu22bl_window_full = False
                mimu4844_window_full = False
    # Once the execution has been stopped, the overall statistics regarding the classifications and repetitions are printed
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
                
                

if __name__ == '__main__':
    # Initialise queues
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()
    # Create instances of the class used to record data
    mimu22bl = Mimu22bl(('COM7', 115200), queue1)
    mimu4844 = Mimu4844(('COM6', 460800), queue2)
    # Defines the multi-processes for recording the data and running the model
    p1 = multiprocessing.Process(name='p1', target=mimu22bl.run)
    p2 = multiprocessing.Process(name='p2', target=mimu4844.run)
    p3 = multiprocessing.Process(name='p3', target=classify, args=(queue1, queue2, 2))
    # Starts the multi-processes
    p1.start()
    p2.start()
    p3.start()