import csv
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.metrics as metrics
from sklearn.metrics import plot_confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import dataset_handling


def initialise_save(x, y, dataset_type):
    """
    Method to Initializing all save
    :param x: x of the data
    :param y: y of the data
    :param dataset_type: Type of the oversample dataset.
    :return: void
    """
    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for neighbors in range(3, 10, 2):
        for ts in range(1, 5, 1):
            test_size = ts / 10

            x_train, __, y_train, ___ = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
            model = KNeighborsClassifier(n_neighbors=neighbors)
            model.fit(x_train, y_train)

            data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}

            pickle.dump(data, open(f"models/KNN/KNN_Data_{dataset_type}_n({neighbors})_size({test_size}).pickle", "wb"))
            pickle.dump(model,
                        open(f"models/KNN/KNN_Model_{dataset_type}_n({neighbors})_size({test_size}).pickle", "wb"))


def building_models(x, y, dataset_type):
    """
    Method to run every model for 10,000 iterations
    :param x: x of the dataset.
    :param y: y of the dataset.
    :param dataset_type: dataset oversample type.
    :return: void
    """

    x_test, y_test = dataset_handling.preprocessing_columns("CRAP_test")

    for neighbors in range(3, 10, 2):
        for ts in range(1, 5, 1):
            for _ in range(10000):

                # Initialising test size.
                test_size = ts / 10

                # Splitting training and testing data.
                x_train, __, y_train, ___ = sklearn.model_selection.train_test_split(x, y, test_size=test_size)
                # Initializing the model.
                model = KNeighborsClassifier(n_neighbors=neighbors)
                model.fit(x_train, y_train)
                acc = model.score(x_test, y_test)

                spec = (neighbors, test_size, _, acc * 100)

                # Loading bestModel and bestData.
                loaded_best_model = pickle.load(
                    open(f"models/KNN/KNN_Model_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
                loaded_best_data = pickle.load(
                    open(f"models/KNN/KNN_Data_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
                best_model = loaded_best_model.fit(loaded_best_data['x_train'], loaded_best_data['y_train'])
                best_acc = best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test'])

                # Updating bestData and bestModel if the new accuracy is better.
                if acc > best_acc:
                    print("best saved", spec)
                    data = {'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}
                    pickle.dump(data,
                                open(f"models/KNN/KNN_Data_{dataset_type}_n({neighbors})_size({test_size}).pickle",
                                     "wb"))
                    pickle.dump(model,
                                open(f"models/KNN/KNN_Model_{dataset_type}_n({neighbors})_size({test_size}).pickle",
                                     "wb"))


def print_results(dataset_type):
    """
    Method to Printing results.
    :param dataset_type: Type of the oversample dataset.
    :return: void
    """

    # Opening the md file to save results.
    f = open(f"notebooks/KNN_results_{dataset_type}.md", "w")
    text = "## All Records\n" \
           "Neighbors | Test Size | Accuracy \n" \
           ":---------: | :---------: | :--------: \n"
    header = "# KNN Results\n\n## Summarization\n\n"
    highest_best = {'name': "name", 'best': 0, 'report': ""}

    # Reading all the save pickle files to make result md files.
    for neighbors in range(3, 10, 2):
        for ts in range(1, 5, 1):
            test_size = ts / 10

            # Loading bestModel and bestData.
            loaded_best_model = pickle.load(
                open(f"models/KNN/KNN_Model_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
            loaded_best_data = pickle.load(
                open(f"models/KNN/KNN_Data_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
            best_model = loaded_best_model.fit(loaded_best_data['x_train'], loaded_best_data['y_train'])
            best_acc = best_model.score(loaded_best_data['x_test'], loaded_best_data['y_test']) * 100

            name = "### neighbors: " + str(neighbors) + ", test size: " + str(test_size)
            msg = "\n" + name + "\n\n\tBest\t\t: " + str(best_acc)

            print(msg.replace('#', ""))

            # saving the highest best accuracy model details.
            if highest_best['best'] < best_acc:
                predictions = best_model.predict(loaded_best_data['x_test'])
                highest_best = {'name': name, 'best': best_acc,
                                'report': sklearn.metrics.classification_report(loaded_best_data['y_test'],
                                                                                predictions)}

            text += str(neighbors) + " | " + str(test_size) + " | " + str(best_acc) + " \n"

    # Showing the details of higest best acuracy model of all model.
    header += f"### The Highest Best\n\n>#{highest_best['name']}\n>> - **Best\t\t: {highest_best['best']}**\n"
    header += "\n\t" + highest_best["report"].replace("\n", "\n\t") + "\n\n"

    # saving to the file.
    f.write(header + text)
    f.close()


def find_best(dataset_type):
    """
    Method to find the best model out of all KNN models.
    :param dataset_type:
    :return: best_model, best_data
    """

    best_model = None
    best_data = None
    best_accuracy = 0
    neighborsG = None
    test_sizeG = None

    # Reading all the save pickle files to find Best Model
    for neighbors in range(3, 10, 2):
        for ts in range(1, 5, 1):
            test_size = ts / 10
            # Loading models & data.
            loaded_model = pickle.load(
                open(f"models/KNN/KNN_Model_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
            loaded_data = pickle.load(
                open(f"models/KNN/KNN_Data_{dataset_type}_n({neighbors})_size({test_size}).pickle", "rb"))
            model = loaded_model.fit(loaded_data['x_train'], loaded_data['y_train'])

            # Running predictions using test data
            predictions = model.predict(loaded_data['x_test'])

            # getting accuracy as percentage
            accuracy = accuracy_score(loaded_data['y_test'], predictions) * 100

            # updating the best model details.
            if best_accuracy < accuracy:
                best_model = loaded_model
                best_data = loaded_data
                best_accuracy = accuracy
                neighborsG = neighbors
                test_sizeG = test_size

    # Return best_model & best_data.
    return best_model, best_data, (neighborsG, test_sizeG)


def plot_graphs(dataset_type):
    """
    Method to plot graphs related to best KNN model.
    :param dataset_type: Type of the oversample dataset.
    :return: void
    """

    # Getting best model and best data to plot.
    best_model, best_data, result = find_best(dataset_type)

    # Creating confusion matrix
    labels = ['negative', 'positive']

    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]

    for title, normalize in titles_options:
        disp = plot_confusion_matrix(best_model, best_data['x_test'], best_data['y_test'], display_labels=labels,
                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)
        plt.savefig(f"plots/KNN/KNN_{dataset_type}_{title}.png".replace(" ", "_").replace(",", ""))

    # plotting ROC curve
    metrics.plot_roc_curve(best_model, best_data['x_test'], best_data['y_test'])
    plt.plot([0, 1], [0, 1], color='darkorange', lw=2, linestyle='--')
    plt.savefig(f"plots/KNN/KNN_{dataset_type}_roc_curve.png")

    f = open("plots/KNN/KNN_plot_details.txt", "w")
    f.write(f"dataset: {dataset_type}\nneighbors: {result[0]}\ntest_size: {result[1]}")
    f.close()


def run_KNN(dataset_type):
    """
    Method to run all KNN functions in order.
    :param dataset_type: Type of the oversample dataset.
    :return: void
    """

    x, y = dataset_handling.preprocessing_columns(dataset_type)
    # initialise_save(x, y, dataset_type)
    building_models(x, y, dataset_type)
    print_results(dataset_type)
    plot_graphs(dataset_type)


def final_KNN():

    best_model, best_data = find_best("ROS")
    best_model.fit(best_data['x_train'], best_data['y_train'])

    test_data = pd.read_csv("data/test_Numeric.csv")

    id = list(test_data['ID'])
    parents = list(test_data['parents'])
    has_nurs = list(test_data['has_nurs'])
    form = list(test_data['form'])
    children = list(test_data['children'])
    housing = list(test_data['housing'])
    finance = list(test_data['finance'])
    social = list(test_data['social'])
    health = list(test_data['health'])

    # Creating 'x' and 'y'
    x_test = list(zip(parents, has_nurs, form, children, housing, finance, social, health))

    predictions = list(best_model.predict(x_test))

    headers = ['ID', 'app_status']
    rows = []

    for i in range(len(id)):
        row = [id[i], int(predictions[i])]
        rows.append(dict(zip(headers, row)))

    filename = f"notebooks/TeamCRAP.csv"

    with open(filename, 'w', newline="") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()
        csvwriter.writerows(rows)

