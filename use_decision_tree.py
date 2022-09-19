from iris_dataset_clean import *
from framework_implementation import *
import os

path = input("Path for the Graphviz bin carpet in your computer (Example: C:/Program Files/Graphviz/bin/): ")
os.environ["PATH"] += os.pathsep + path

option = 1

def print_menu():
    print("""Options:
    1. Predict data
    2. Plot Decision Tree
    3. Print Metrics
    4. Use Validation Set
    5. Print Metrics with validation set
    6. Print Bias and Variance
    7. Exit
    """)
    return int(input("Select and option: "))

model = NewDecisionTreeModel()
model.fit(X_train, y_train)

print("--- NOTE: Iris Dataset already loaded and fitted in model for usage ---\n")

y_pred = []
y_pred_val = []

while True:
    option = print_menu()
    if option == 1:
        y_pred = model.predict(X_test)
        print("Predictions:", y_pred)
        print("Predicted data with Success!")
    elif option == 2:
        model.plot_graph(feature_names = X_train.columns, class_names = species)
    elif option == 3:
        if not len(y_pred) == 0:
            model.print_metrics(y_pred, y_test, class_names = species)
        else:
            print("No predictions to load metrics from")
    elif option == 4:
        if not len(y_pred) == 0:
            y_pred_val = model.predict(X_val)
            print("Predictions:", y_pred_val)
            print("Predicted validation data with Success!")
        else:
            print("First you've got to predict the test data")
    elif option == 5:
        if not len(y_pred) == 0 and not len(y_pred_val) == 0:
            model.print_metrics(y_pred, y_test, class_names = species)
            print("-"*30)
            model.print_metrics(y_pred_val, y_val, class_names = species)
        else:
            print("First you've got to predict the test data")
        
    elif option == 6:
        model.print_bias(X_train.values, y_train.values, X_test.values, y_test.values)
    elif option == 7:
        break
    else:
        print("That's not a valid option")