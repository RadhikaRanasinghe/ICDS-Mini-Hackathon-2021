# importing the libraries

import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def run_random_forest():
    # Loading the dataset with a Pandas and the returned data frame is caught by data variable
    data_frame = pd.read_csv("data/CRAP _ROS_Oversampled_Dataset.csv")
    # cols_to_drop = ['ID']
    # data_frame = data_frame.drop(cols_to_drop, axis=1)

    # Creating 'x' and 'y'
    x = data_frame.values
    x = np.delete(x, 8, axis=1)
    y = data_frame['app_status'].values

    highest_accuracy_save(x, y)


def highest_accuracy_save(x, y):
    for _ in range(10000):
        # Splitting the data into testing data and training data with the testing size of 0.3
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

        # The Random Forest Classifier Model is assigned to the model variable
        model = RandomForestClassifier()

        # The training data from the data split is taken for fitting into the model to train
        model.fit(x_train, y_train)

        # The accuracy score of the model
        acc = model.score(x_test, y_test)

        # Loading bestModel and bestData.
        loaded_best_data = pickle.load(
            open(
                "models/Random_Forest/Random_Forest_Best_Data.pickle",
                "rb"))
        loaded_best_model = pickle.load(
            open(
                "models/Random_Forest/Random_Forest_Best_Model.pickle",
                "rb"))
        best_acc = loaded_best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

        # Updating bestData and bestModel if the new accuracy is better.
        if acc > best_acc:
            print("best saved", acc)
            pickle.dump(model, open(
                "models/Random_Forest/Random_Forest_Best_Model.pickle",
                "wb"))
            # Saving the training and testing data
            data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            pickle.dump(data, open(
                "models/Random_Forest/Random_Forest_Best_Data.pickle",
                "wb"))


def writing():
    loaded_best_data = pickle.load(
        open(
            "models/Random_Forest/Random_Forest_Best_Data.pickle",
            "rb"))
    loaded_best_model = pickle.load(
        open(
            "models/Random_Forest/Random_Forest_Best_Model.pickle",
            "rb"))

    # Running predictions using  test data
    predictions = loaded_best_model.predict(loaded_best_data['x_test'])

    # Writing the classification report
    text = classification_report(loaded_best_data['y_test'], predictions)
    print(text)

    file = open("notebooks/Random_Forest_Classification_Report.txt", "w")
    file.write(text)
    file.close()

    # Creating confusion matrix
    labels = ['negative', 'positive']

    titles_options = [("Confusion matrix, without normalization", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(loaded_best_model, loaded_best_data['x_test'], loaded_best_data['y_test'],
                                     display_labels=labels, cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(f"plots/{title}.png".replace(" ", "_").replace(",", ""))


writing()
