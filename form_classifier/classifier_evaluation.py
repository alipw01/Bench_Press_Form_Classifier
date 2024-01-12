from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class ModelEvaluator:
    def __init__(self, y_pred, y_test):
        """
        param y_pred: the predicted classes
        param y_test: the actual classes
        """
        self.y_pred = y_pred
        self.y_test = y_test
    
    def evaluate(self):
        """
        Function for evaluating the performance of the predictions
        return: the calculated accuracy, precision, recall, f1 and confusion matrix
        """
        # Calculate evaluation metrics
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average='macro')
        recall = recall_score(self.y_test, self.y_pred, average='macro')
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Print evaluation metrics
        print("Accuracy: {:.3f}".format(accuracy))
        print("Precision: {:.3f}".format(precision))
        print("Recall: {:.3f}".format(recall))
        print("F1 Score: {:.3f}".format(f1))
        print("Confusion Matrix:")
        print(cm)
        
        return accuracy, precision, recall, f1, cm