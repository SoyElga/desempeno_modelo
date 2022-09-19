import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error, accuracy_score, confusion_matrix, classification_report
from mlxtend.evaluate import bias_variance_decomp

class NewDecisionTreeModel:

    def __init__(self):
        self.clf = DecisionTreeClassifier(random_state = 0)
    
    def fit(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    def predict(self, X_test):
        self.prediction = self.clf.predict(X_test)
        return self.prediction
    
    def plot_graph(self, feature_names, class_names):
        dot_data = export_graphviz(self.clf, out_file=None, feature_names=feature_names, class_names=class_names, filled=True, rounded=True, special_characters=True) 
        graph = graphviz.Source(dot_data)
        graph.render(view=True)

    def print_metrics(self, y_pred, y_test, class_names):
        print("Metrics:")
        print("\tAccuracy:",accuracy_score(y_test, y_pred))
        print("\tR^2 : ", r2_score(y_test, y_pred))
        print("\tMAE :", mean_absolute_error(y_test,y_pred))
        print("\tRMSE:",np.sqrt(mean_squared_error(y_test, y_pred)))
        print("\nConfusion Matrix")
        print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns = class_names))
        print("\nClassification report")
        print(classification_report(y_test, y_pred))

    def print_bias(self, X_train, y_train, X_test, y_test):
        mse, bias, var = bias_variance_decomp(self.clf, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=42)
        print("MSE:", mse)
        print("BIAS:", bias)
        print("VAR:", var)

    


    
    

