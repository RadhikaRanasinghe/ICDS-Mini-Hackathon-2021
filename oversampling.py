from collections import Counter
from imblearn.over_sampling import ADASYN, RandomOverSampler
import csv
import numpy as np
import pandas as pd
import sklearn
import pickle
import sklearn.preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


def preprocessing_columns(data):
    # iden = list(data['ID'])
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


def write_data(x, y):
    headers = ['parents', 'has_nurs', 'form', 'children', 'housing','finance', 'social', 'health', 'app_status']
    rows = []

    for i in range(len(x)):
        row = list(x[i])
        row.append(int(y[i]))
        rows.append(dict(zip(headers, row)))

    # filename = "data/ICDS_Oversampled_Dataset.csv"
    filename = "data/ICDS_ROS_Oversampled_Dataset.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
        csvwriter.writeheader()
        csvwriter.writerows(rows)


def oversample_data():
    data = pd.read_csv('data/ICDS_Numeric_Dataset.csv')
    data.drop("ID", axis=1, inplace=True)

    x, y = preprocessing_columns(data)

    # smote_nc = SMOTENC(categorical_features=[0, 1, 2, 3, 4, 5, 6, 7], random_state=0)
    # X_resampled, y_resampled = smote_nc.fit_resample(x, y)
    # print(sorted(Counter(y_resampled).items()))

    # x_resampled, y_resampled = ADASYN().fit_resample(x, y)
    # print(sorted(Counter(y_resampled).items()))

    ros = RandomOverSampler()
    x_resampled, y_resampled = ros.fit_resample(x, y)
    print(sorted(Counter(y_resampled).items()))

    write_data(x_resampled, y_resampled)


# preprocessing("data/train_data.csv")
oversample_data()

# data = pd.read_csv('data/ICDS_Oversampled_Dataset.csv')
# sns.countplot(x="app_status", data=data)
# plt.show()
