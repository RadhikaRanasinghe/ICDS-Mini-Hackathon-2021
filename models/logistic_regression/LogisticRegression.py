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
from sklearn.metrics import accuracy_score


def logisticRegression():
    highest = 0
    for x in range(1000):
        train_data = pd.read_csv("ICDS_ROS_Oversampled_Dataset.csv", sep=",")
        test_data = pd.read_csv("ICDS_Numeric_Dataset_Test.csv", sep=",")
        X = train_data.drop("app_status", axis=1)
        y = train_data["app_status"]

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        model = LogisticRegression(solver='lbfgs', max_iter=20000)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions) * 100
        # print(accuracy)
        print("round : " + str(x))
        if accuracy > highest:
            highest = accuracy

    print(accuracy)


# preprocessing("test_data.csv")

logisticRegression()
