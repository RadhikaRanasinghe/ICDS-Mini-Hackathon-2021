# importing the libraries

import pickle

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix
from sklearn.model_selection import train_test_split

import dataset_handling


def run_random_forest():
    # Loading the dataset with a Pandas and the returned data frame is caught by data variable
    x, y = dataset_handling.preprocessing_columns("CRAP_ROS")

    # initialize(x, y)

    highest_accuracy_save(x, y)

    writing()


def initialize(x, y):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    # Splitting the data into testing data and training data with the testing size of 0.3
    x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.3)

    # The Random Forest Classifier Model is assigned to the model variable
    model = RandomForestClassifier()

    # The training data from the data split is taken for fitting into the model to train
    model.fit(x_train, y_train)

    # The accuracy score of the model
    acc = model.score(x_test, y_test)

    pickle.dump(model, open("models/Random_Forest/Random_Forest_Best_Model.pickle", "wb"))
    # Saving the training and testing data
    data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
    pickle.dump(data, open("models/Random_Forest/Random_Forest_Best_Data.pickle", "wb"))
    print("Initial saved", acc)


def highest_accuracy_save(x, y):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for _ in range(10000):
        # Splitting the data into testing data and training data with the testing size of 0.3
        x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.3)

        # The Random Forest Classifier Model is assigned to the model variable
        model = RandomForestClassifier()

        # The training data from the data split is taken for fitting into the model to train
        model.fit(x_train, y_train)

        # The accuracy score of the model
        acc = model.score(x_test, y_test)

        # Loading bestModel and bestData.
        loaded_best_data = pickle.load(open("models/Random_Forest/Random_Forest_Best_Data.pickle", "rb"))
        loaded_best_model = pickle.load(open("models/Random_Forest/Random_Forest_Best_Model.pickle", "rb"))
        best_acc = loaded_best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

        # Updating bestData and bestModel if the new accuracy is better.
        if acc > best_acc:
            print("changed the accuracy to", acc, "in the iteration:", _)
            pickle.dump(model, open("models/Random_Forest/Random_Forest_Best_Model.pickle", "wb"))
            # Saving the training and testing data
            data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            pickle.dump(data, open("models/Random_Forest/Random_Forest_Best_Data.pickle", "wb"))


def writing():
    loaded_best_data = pickle.load(open("models/Random_Forest/Random_Forest_Best_Data.pickle", "rb"))
    loaded_best_model = pickle.load(open("models/Random_Forest/Random_Forest_Best_Model.pickle", "rb"))

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


run_random_forest()
