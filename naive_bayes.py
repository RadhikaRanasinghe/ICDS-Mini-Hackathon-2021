import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from dataset_handling import preprocessing_columns


def naive_bayes():
    dataset_type = "CRAP_ROS"
    x, y = preprocessing_columns(dataset_type)
    x_test, y_test = preprocessing_columns("CRAP_test")

    # Naive Bayes Classifier
    lowest_model = 0
    highest_model = 0
    lowest_score = 0
    highest_score = 0
    high_count = 0

    for i in range(10000):
        x_train, _, y_train, __ = train_test_split(x, y, test_size=0.1, random_state=None)
        model = GaussianNB()  # Naive Bayes model
        model.fit(x_train, y_train)  # fit model - train the model
        predicted = model.predict(x_test)  # Y_pred
        score = metrics.accuracy_score(y_test, predicted)
        print(score * 100)

        # first iteration
        if i == 0:
            lowest_model = model
            highest_model = model
            lowest_score = score
            highest_score = score
        else:
            if score < lowest_score:
                highest_score = score
                lowest_model = model

                print(metrics.classification_report(y_test, predicted))
                print(metrics.confusion_matrix(y_test, predicted))

                file = open(f"notebooks/NB_{dataset_type}_WorstModels.txt", "a")
                file.write(metrics.classification_report(y_test, predicted) + "\nAccuracy : " + str(lowest_score) + "\n")
                file.close()

            elif score > highest_score:
                highest_score = score
                highest_model = model
                high_count += 1

                print(metrics.classification_report(y_test, predicted))
                print(metrics.confusion_matrix(y_test, predicted))

                # Saving models as pickle files
                with open(f"models/Naives_Bayes/NB_{dataset_type}_BestModel_%s.pickle" % high_count, "wb") as bestModel:
                    pickle.dump(highest_model, bestModel)

                file = open(f"notebooks/NB_{dataset_type}_results.txt", "a")
                file.write(metrics.classification_report(y_test, predicted) + "\nAccuracy : " + str(highest_score) + "\n")
                file.close()

    # Saving models as pickle files
    with open(f"models/Naives_Bayes/NB_{dataset_type}_WorstModel.pickle", "wb") as worstModel:
        pickle.dump(lowest_model, worstModel)


def nb_confusion_matrix(dataset_type):
    x_test, y_test = preprocessing_columns(dataset_type)

    # Loading models from pickle files
    pickle_in = open(f'models/Naives_Bayes/NB_CRAP_ROS_1.pickle', "rb")
    loaded_model = pickle.load(pickle_in)

    # make prediction using loaded model
    new_pred = loaded_model.predict(x_test)
    new_score = metrics.accuracy_score(y_test, new_pred)

    # plot non-normalized confusion matrix
    options = [("Confusion matrix, without normalization", None),
               ("Normalized confusion matrix", 'true')]
    for title, normalize in options:
        plot = plot_confusion_matrix(loaded_model, x_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
        plot.ax_.set_title(title)
        print(plot.confusion_matrix)

    plt.show()
    print(new_score * 100)


def nb_roc_curve(dataset_type):
    x_test, y_test = preprocessing_columns(dataset_type)

    # Loading models from pickle files
    pickle_in = open(f'models/Naives_Bayes/NB_CRAP_ROS_1.pickle', "rb")
    loaded_model = pickle.load(pickle_in)

    # plot roc curve
    metrics.plot_roc_curve(loaded_model, x_test, y_test)
    plt.show()


def run_naive_bayes():
    naive_bayes()
    nb_confusion_matrix("CRAP_test")
    nb_roc_curve("CRAP_test")
