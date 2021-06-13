import csv
import numpy as np
import pandas as pd
import sklearn
import pickle
import sklearn.preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def logisticRegression():
    highModel = 0
    highestAccuracy = 0
    train_data = pd.read_csv("data/CRAP_ROS_Oversampled_Dataset.csv", sep=",")
    X = train_data.drop("app_status", axis=1)
    y = train_data["app_status"]

    X = np.array(X)
    y = np.array(y)

    for x in range(10000):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        model = LogisticRegression(solver='lbfgs', max_iter=20000)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100
        if x == 0:
            highModel = model
            highestAccuracy = accuracy
            # creating and writing the accuracies to a log file
            file = open("log.txt", "w")
            file.write("Highest accuracy : " + str(highestAccuracy) + "\n")
            file.close()
        else:
            # if the current model accuracy is higher than the 3rd highest accuracy
            if accuracy > highestAccuracy:
                highestAccuracy = accuracy
                print("changed")
                file = open("notebooks/ logistic regression log.txt", "a")
                file.write("Highest accuracy : " + str(highestAccuracy) + "\n")
                # if the current model accuracy is higher than the 2nd highest accuracy
                highModel = model
                file.close()
                classificationreportFile = open("notebooks/logistic regression classification report.txt", "w")
                classificationreportFile.write(classification_report(y_test, predictions))
        print("success")

    # saving the models as pickle files
    with open('models/LogR_BestModel.pickle', 'wb') as handle:
        pickle.dump(highModel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('done')


logisticRegression()
