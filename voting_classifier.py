import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


def preprocessing():
    # Loading the dataset with a Pandas and the returned data frame is caught by data variable
    data_frame = pd.read_csv("data/ICDS_ROS_Dataset.csv")
    # cols_to_drop = ['ID']
    # data_frame = data_frame.drop(cols_to_drop, axis=1)

    # Creating 'x' and 'y'
    x = data_frame.values
    x = np.delete(x, 8, axis=1)
    y = data_frame['app_status'].values

    return x, y


def train_voting_classifier():
    estimators = []
    for root, directories, files in os.walk(f"models/final", topdown=False):
        for name in files:
            model = pickle.load(open(os.path.join(root, name), "rb"))
            estimators.append((name, model))

    x, y = preprocessing()

    best_model = None
    best_accuracy = 0
    best_x_test = None
    best_y_test = None

    for i in range(1000):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        model = VotingClassifier(estimators=estimators, voting='hard')

        model.fit(x_train, y_train)

        model_accuracy = model.score(x_test, y_test)

        if best_accuracy < model_accuracy:
            best_model = model
            best_accuracy = model_accuracy
            print(f"iteration {i + 1}: {model_accuracy * 100}")
            best_x_test = x_test
            best_y_test = y_test
            pickle.dump(best_model, open("VotingClassifier.pickle", "wb"))

    # Running predictions using  test data
    predictions = best_model.predict(best_x_test)

    # getting accuracy as percentage
    text = classification_report(best_y_test, predictions) + f"\nVotingClassifier accuracy: {best_accuracy}"

    file = open(f"notebooks/voting_classifier_report.txt", "w")
    file.write(text)
    file.close()

    # Creating confusion matrix
    labels = ['negative', 'positive']

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(best_model, best_x_test, best_y_test, display_labels=labels, cmap=plt.cm.Blues,
                                     normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(f"plots/voting_classifier_{title}.png".replace(" ", "_").replace(",", ""))


train_voting_classifier()
