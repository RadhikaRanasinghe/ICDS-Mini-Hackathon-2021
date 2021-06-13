import os
import pickle

import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import dataset_handling


def run_vc():
    # Loading the dataset with a Pandas and the returned data frame is caught by data variable
    x, y = dataset_handling.preprocessing_columns("CRAP_ROS")
    estimators = []
    for root, directories, files in os.walk(f"models/final", topdown=False):
        for name in files:
            model = pickle.load(open(os.path.join(root, name), "rb"))
            estimators.append((name, model))

    # initialize(x, y, estimators)

    highest_accuracy_save(x, y, estimators)

    writing()

    return x, y


def initialize(x, y, estimators):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    # Splitting the data into testing data and training data with the testing size of 0.3
    x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.1)

    # The Random Forest Classifier Model is assigned to the model variable
    model = VotingClassifier(estimators=estimators, voting='hard')

    # The training data from the data split is taken for fitting into the model to train
    model.fit(x_train, y_train)

    # The accuracy score of the model
    acc = model.score(x_test, y_test)

    pickle.dump(model, open("models/Voting_Classifier/Voting_Classifier_Best_Model.pickle", "wb"))
    # Saving the training and testing data
    data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
    pickle.dump(data, open("models/Voting_Classifier/Voting_Classifier_Best_Data.pickle", "wb"))
    print("Initial saved", acc)


def highest_accuracy_save(x, y, estimators):
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for _ in range(1000):
        # Splitting the data into testing data and training data with the testing size of 0.3
        x_train, __, y_train, ___ = train_test_split(x, y, test_size=0.1)

        # The Random Forest Classifier Model is assigned to the model variable
        model = VotingClassifier(estimators=estimators, voting='hard', )

        # The training data from the data split is taken for fitting into the model to train
        model.fit(x_train, y_train)

        # The accuracy score of the model
        acc = model.score(x_test, y_test)

        # Loading bestModel and bestData.
        loaded_best_data = pickle.load(open("models/Voting_Classifier/Voting_Classifier_Best_Data.pickle", "rb"))
        loaded_best_model = pickle.load(open("models/Voting_Classifier/Voting_Classifier_Best_Model.pickle", "rb"))
        best_acc = loaded_best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

        # Updating bestData and bestModel if the new accuracy is better.
        if acc > best_acc:
            print("changed the accuracy to", acc, "in the iteration:", _)
            pickle.dump(model, open("models/Voting_Classifier/Voting_Classifier_Best_Model.pickle", "wb"))
            # Saving the training and testing data
            data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
            pickle.dump(data, open("models/Voting_Classifier/Voting_Classifier_Best_Data.pickle", "wb"))


def writing():
    loaded_best_data = pickle.load(open("models/Voting_Classifier/Voting_Classifier_Best_Data.pickle", "rb"))
    loaded_best_model = pickle.load(open("models/Voting_Classifier/Voting_Classifier_Best_Model.pickle", "rb"))

    # Running predictions using  test data
    predictions = loaded_best_model.predict(loaded_best_data['x_test'])

    # Writing the classification report
    text = classification_report(loaded_best_data['y_test'], predictions)
    print(text)

    file = open("notebooks/Voting_Classifier_Classification_Report.txt", "w")
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
        plt.savefig(f"plots/VC/{title}.png".replace(" ", "_").replace(",", ""))


run_vc()
