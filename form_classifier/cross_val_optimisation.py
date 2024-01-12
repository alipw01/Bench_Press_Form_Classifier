import pickle
from tqdm import tqdm
from preprocessing import *
from feature_extraction import *
from data_splitting import *
from classification import *
from classifier_evaluation import *
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    # Defines hyperparameter grids for each classifier
    logreg_param_grid = {
        'penalty': ['l2'],
        'C': [0.1, 1, 10],
        'solver': ['newton-cg', 'lbfgs', 'liblinear'],
        'max_iter': [1000, 5000],
        'multi_class': ['ovr', 'auto']
    }

    mlp_param_grid = {
        'activation': ['logistic', 'tanh', 'relu'],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'max_iter': [5000],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'random_state': [42]
    }

    svc_param_grid = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': [2, 3, 4],
        'probability': [True]
    }

    knn_param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [20, 30, 40],
        'p': [1, 2]
    }

    gnb_param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }

    rf_param_grid = {
        'n_estimators': [100, 200],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'random_state': [42]
    }

    # Splits data
    data_splitter = DataSplitter('./data/cleaned_data/')
    train, test = data_splitter.split_data_rolling_window()

    # Defines the potential feature extraction window sizes in seconds
    window_sizes = [1.5,2,2.5]

    # Creates list for storing the best hyperparameters and performance for each window size
    all_data = []

    # Iterates through each potential window size
    for window_size in window_sizes:
        training_features = []
        testing_features = []
        # Pre-processes and extracts features for training set
        for pair in tqdm(train, desc="Preprocessing Data and Extracting Features", unit="file pair"):
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
        for pair in tqdm(test, desc="Preprocessing Data and Extracting Features", unit="file pair"):
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
        training_features = pd.concat(training_features)
        testing_features = pd.concat(testing_features)

        # Creates training and testing values
        print('Training Model...')
        x_train = training_features.drop('label', axis=1)
        y_train = training_features['label']
        x_test = testing_features.drop('label', axis=1)
        y_test = testing_features['label']

        # Create a GridSearchCV object for each estimator
        logreg_grid = GridSearchCV(LogisticRegression(), logreg_param_grid)
        mlp_grid = GridSearchCV(MLPClassifier(), mlp_param_grid)
        svc_grid = GridSearchCV(SVC(), svc_param_grid)
        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid)
        gnb_grid = GridSearchCV(GaussianNB(), gnb_param_grid)
        rf_grid = GridSearchCV(RandomForestClassifier(), rf_param_grid)

        # Fit the GridSearchCV objects to the data
        print('logreg CV...')
        logreg_grid.fit(x_train, y_train)
        print('mlp CV...')
        mlp_grid.fit(x_train, y_train)
        print('svc CV...')
        svc_grid.fit(x_train, y_train)
        print('knn CV...')
        knn_grid.fit(x_train, y_train)
        print('gnb CV...')
        gnb_grid.fit(x_train, y_train)
        print('rf CV...')
        rf_grid.fit(x_train, y_train)
        print('fitted')

        # Get the best hyperparameters for each classifier
        logreg_best = logreg_grid.best_params_
        mlp_best = mlp_grid.best_params_
        svc_best = svc_grid.best_params_
        knn_best = knn_grid.best_params_
        gnb_best = gnb_grid.best_params_
        rf_best = rf_grid.best_params_
        # Creates list of all optimised classifier hyperparameters
        dicts = [logreg_best, mlp_best, svc_best, knn_best, gnb_best, rf_best]
        # Trains model using optimal hyperparameters and calculates mean cross-validated accuracy
        model = ClassifierCrossValidation(logreg_args=logreg_best, 
                                            mlp_args=mlp_best, 
                                            svc_args=svc_best, 
                                            knn_args=knn_best, 
                                            gnb_args=gnb_best, 
                                            rf_args=rf_best)
        cross_val_mean = model.train(x_train, y_train)
        # Predicts classes of testing set
        print('Predicting Test Data...')
        y_pred = model.predict(x_test)
        # Evaluates performance of predictions
        evaluator = ModelEvaluator(y_pred, y_test)
        accuracy, precision, recall, f1, cm = evaluator.evaluate()
        # Append information regarding the optimal setup for the window size and its performance to all_data list
        all_data.append([dicts, cross_val_mean, accuracy, precision, recall, f1, window_size])
    
    # Prints the best performance and setup for each window size
    print('Best performance for each window size: ', all_data)
    # Enables the saving of the trained model for future use
    save = raw_input('Would you like to save the Model (Y/N)? ')
    if save.upper() == 'Y':
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    elif save.upper() == 'N':
        pass
    else:
        print('Invalid input. Please enter Y or N.')