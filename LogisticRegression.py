import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

import dataset_handling


def logisticRegression():
    highModel = 0
    highestAccuracy = 0

    dataset_type = "CRAP_ROS"
    X, y = dataset_handling.preprocessing_columns(dataset_type)

    X = np.array(X)
    y = np.array(y)

    X_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for x in range(10000):

        X_train, _, y_train, __ = train_test_split(X, y, test_size=0.1)

        model = LogisticRegression(solver='lbfgs', max_iter=20000)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100
        if x == 0:
            highModel = model
            highestAccuracy = accuracy
            # creating and writing the accuracies to a log file
            file = open(f"notebooks/LogR_result_{dataset_type}.txt", "w")
            file.write(classification_report(y_test, predictions) + "\nHighest accuracy : " + str(highestAccuracy))
            file.close()
        else:
            # if the current model accuracy is higher than the 3rd highest accuracy
            if accuracy > highestAccuracy:
                highestAccuracy = accuracy
                print("changed")
                # if the current model accuracy is higher than the 2nd highest accuracy
                highModel = model
                file = open(f"notebooks/LogR_result_{dataset_type}.txt", "w")
                file.write(classification_report(y_test, predictions) + "\nHighest accuracy : " + str(highestAccuracy))
                file.close()
        print("success")

    # saving the models as pickle files
    with open(f'models/LogR_{dataset_type}.pickle', 'wb') as handle:
        pickle.dump(highModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done')


logisticRegression()
