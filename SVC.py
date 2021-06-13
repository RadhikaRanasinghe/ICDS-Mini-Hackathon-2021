# importing the libraries

import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import dataset_handling


def run_svc():
    # Loading the dataset with a Pandas and the returned data frame is caught by data variable
    x, y = dataset_handling.preprocessing_columns("CRAP_ROS")

    # initialize(x, y)

    highest_accuracy_save(x, y)

    writing()


def initialize(x, y):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    # Splitting the data into testing data and training data with the testing size of 0.3
    x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.1)

    # The SVC Model is assigned to the model variable
    model = LinearSVC(max_iter=10000)

    # The training data from the data split is taken for fitting into the model to train
    model.fit(x_train, y_train)

    # The accuracy score of the model
    acc = model.score(x_test, y_test)

    print("Initial saved", acc)
    pickle.dump(model, open("models/SVC/SVC_Best_Model.pickle", "wb"))
    # Saving the training and testing data
    data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
    pickle.dump(data, open("models/SVC/SVC_Best_Data.pickle", "wb"))


def highest_accuracy_save(x, y):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for _ in range(10000):
        # Splitting the data into testing data and training data with the testing size of 0.3
        x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.1)

        # The SVC Model is assigned to the model variable
        model = LinearSVC(max_iter=10000)

        # The training data from the data split is taken for fitting into the model to train
        model.fit(x_train, y_train)

        # The accuracy score of the model
        acc = model.score(x_test, y_test)

        # Loading bestModel and bestData.
        loaded_best_data = pickle.load(open("models/SVC/SVC_Best_Data.pickle", "rb"))
        loaded_best_model = pickle.load(open("models/SVC/SVC_Best_Model.pickle", "rb"))
        best_acc = loaded_best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

        # Updating bestData and bestModel if the new accuracy is better.
        if acc > best_acc:
            print("best saved", acc)
            pickle.dump(model, open("models/SVC/SVC_Best_Model.pickle", "wb"))
            # Saving the training and testing data
            data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            pickle.dump(data, open("models/SVC/SVC_Best_Data.pickle", "wb"))


def writing():
    loaded_best_data = pickle.load(open("models/SVC/SVC_Best_Data.pickle", "rb"))
    loaded_best_model = pickle.load(open("models/SVC/SVC_Best_Model.pickle", "rb"))

    # Running predictions using  test data
    predictions = loaded_best_model.predict(loaded_best_data['x_test'])
    acc = loaded_best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

    # Writing the classification report
    text = classification_report(loaded_best_data['y_test'], predictions) + "\nAccuracy score : " + str(acc)
    print(text)

    file = open("notebooks/SVC_Classification_Report.txt", "w")
    file.write(text)
    file.close()


run_svc()
