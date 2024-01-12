from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score


class InertialSensorClassifier:
    """
    Class for classifying unseen data
    """
    def __init__(self):
        # Defines each classifier
        self.logreg = LogisticRegression(penalty='l2', multi_class='auto', C=0.1, max_iter=5000, solver='lbfgs')
        self.mlp = MLPClassifier(random_state=42, activation='logistic', max_iter=5000, learning_rate='constant', solver='adam')
        self.svm = SVC(kernel='linear', probability=True, gamma='scale', degree=2)
        self.knn = KNeighborsClassifier(n_neighbors=3, weights='uniform', leaf_size=20, algorithm='auto', p=1)
        self.nb = GaussianNB(var_smoothing=1e-07)
        self.rf = RandomForestClassifier(max_features='sqrt', n_estimators=200, bootstrap=True, criterion='gini', random_state=42)
        # Defines ensemble
        self.ensemble = VotingClassifier(estimators=[
            ('logreg', self.logreg), 
            ('mlp', self.mlp), 
            ('svm', self.svm), 
            ('knn', self.knn), 
            ('nb', self.nb), 
            ('rf', self.rf)], 
            voting='soft', 
            n_jobs=-1)
        
    def train(self, X_train, y_train):
        """
        Function for training the ensemble
        param X_train: the extracted features
        param y_train: the ground truth classes
        """
        self.ensemble.fit(X_train, y_train)
        
    def predict(self, X_test):
        """
        Function for predicting the class of unseen data
        param x: the extracted features
        return: the predicted classes
        """
        return self.ensemble.predict(X_test)


class ClassifierCrossValidation:
    """
    Class for performing cross-validation, allowing for the passing of classifier hyperparameters as arguments and printing the mean cross-validated score upon training
    """
    def __init__(self, logreg_args=None, mlp_args=None, svm_args=None, knn_args=None, gnb_args=None, rf_args=None):
        """
        param logreg_args: the logistic regression hyperparameters
        param mlp_args: the multilayer perceptron hyperparameters
        param svm_args: the support vector machine hyperparameters
        param knn_args: the k-nearest neighbour hyperparameters
        param gnb_args: the gaussian naive bayes hyperparameters
        param rf_args: the random forest hyperparameters
        """
        # Defines each classifier
        self.logreg = LogisticRegression(**(logreg_args or dict(penalty='l2', multi_class='auto', C=0.1, max_iter=5000, solver='lbfgs')))
        self.mlp = MLPClassifier(**(mlp_args or dict(random_state=42, activation='logistic', max_iter=5000, learning_rate='constant', solver='adam')))
        self.svm = SVC(**(svm_args or dict(kernel='linear', probability=True, gamma='scale', degree=2)))
        self.knn = KNeighborsClassifier(**(knn_args or dict(n_neighbors=3, weights='uniform', leaf_size=20, algorithm='auto', p=1)))
        self.nb = GaussianNB(**(gnb_args or dict(var_smoothing=1e-07)))
        self.rf = RandomForestClassifier(**(rf_args or dict(max_features='sqrt', n_estimators=200, bootstrap=True, criterion='gini', random_state=42)))
        # Defines the ensemble classifier
        self.ensemble = VotingClassifier(estimators=[
            ('logreg', self.logreg), 
            ('mlp', self.mlp), 
            ('svm', self.svm), 
            ('knn', self.knn), 
            ('nb', self.nb), 
            ('rf', self.rf)], 
            voting='soft', 
            n_jobs=-1)
        
    def train(self, x, y):
        """
        Function for calculating the mean cross-validated accuracy and training the ensemble
        param x: the extracted features
        param y: the ground truth classes
        return: mean cross-validated accuracy
        """
        scores = cross_val_score(self.ensemble, x, y, cv=10, n_jobs=-1)
        print("Accuracy: {:.2f} (+/- {:.2f})".format(scores.mean(), scores.std() * 2))
        self.ensemble.fit(x, y)
        return scores.mean()
    
    def predict(self, x):
        """
        Function for predicting the class of unseen data
        param x: the extracted features
        return: the predicted classes
        """
        return self.ensemble.predict(x)