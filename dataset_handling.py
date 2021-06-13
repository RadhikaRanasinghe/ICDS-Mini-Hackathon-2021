import csv
from collections import Counter

import pandas as pd
import sklearn
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours


def create_Numeric(dataset_type):
    file_handler = open(f"data/{dataset_type}_data.csv", "r")
    data = pd.read_csv(file_handler, sep=",")
    df = pd.DataFrame(data)  # define dataframe
    file_handler.close()

    # replace the strings with integers in dataframe
    new_df = df.replace(["usual", "pretentious", "great_pret",  # parents
                         "less_proper", "improper", "proper", "critical", "very_crit",  # has_nurs
                         "foster", "incomplete", "complete", "completed",  # form
                         "more",  # children
                         "less_conv", "convenient", "critical",  # housing
                         "convenient", "inconv",  # finance
                         "nonprob", "slightly_prob", "problematic",  # social
                         "not_recom", "recommended", "priority"],  # health
                        [1, 2, 3,  # parents
                         1, 2, 3, 4, 5,  # has_nurs
                         1, 2, 3, 4,  # form
                         4,  # children
                         1, 2, 3,  # housing
                         1, 2,  # finance
                         1, 2, 3,  # social
                         1, 2, 3])  # health

    # change the file to anything if needed :)
    new_df.to_csv(f"data/{dataset_type}_Numeric.csv", index=False)


def create_test_train():
    x, y = preprocessing_columns("train_Numeric")

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    write_data(x_train, y_train, "CRAP_train")
    write_data(x_test, y_test, "CRAP_test")


def preprocessing_columns(dataset_type):
    data = pd.read_csv(f'data/{dataset_type}.csv')

    parents = list(data['parents'])
    has_nurs = list(data['has_nurs'])
    form = list(data['form'])
    children = list(data['children'])
    housing = list(data['housing'])
    finance = list(data['finance'])
    social = list(data['social'])
    health = list(data['health'])
    app_status = list(data['app_status'])

    # Creating 'x' and 'y'
    x = list(zip(parents, has_nurs, form, children, housing, finance, social,
                 health))
    y = list(app_status)

    return x, y


def write_data(x, y, dataset_type):
    headers = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health', 'app_status']
    rows = []

    for i in range(len(x)):
        row = list(x[i])
        row.append(int(y[i]))
        rows.append(dict(zip(headers, row)))

    filename = f"data/{dataset_type}.csv"

    with open(filename, 'w', newline="") as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()
        csvwriter.writerows(rows)


def create_oversample():
    x, y = preprocessing_columns("CRAP_train")

    # ADASYN
    dataset_type = "CRAP_ADASYN"
    x_resampled, y_resampled = ADASYN().fit_resample(x, y)
    print(dataset_type, sorted(Counter(y_resampled).items()))
    write_data(x_resampled, y_resampled, dataset_type)

    # ROS
    dataset_type = "CRAP_ROS"
    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(x, y)
    print(dataset_type, sorted(Counter(y_resampled).items()))
    write_data(x_resampled, y_resampled, dataset_type)


def create_undersample():
    x, y = preprocessing_columns("CRAP_train")

    dataset_type = "CRAP_RENN"
    renn = RepeatedEditedNearestNeighbours()
    X_resampled, y_resampled = renn.fit_resample(x, y)
    print(dataset_type, sorted(Counter(y_resampled).items()))
    write_data(X_resampled, y_resampled, dataset_type)
