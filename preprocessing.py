import csv
import numpy as np
import pandas as pd
import sklearn
import pickle
import sklearn.preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


def preprocessing(dataset):
    file_handler = open(dataset, "r")
    data = pd.read_csv(file_handler, sep=",")
    df = pd.DataFrame(data)  # define dataframe
    file_handler.close()

    print(data)

    # replace the strings with integers in dataframe
    new_df = df.replace(["usual", "pretentious", "great_pret",  # parents
                         "less_proper", "very_crit", "proper", "improper",  # has_nurs
                         "complete", "incomplete", "completed", "foster",  # form
                         "more",  # children
                         "convenient", "critical", "less_conv",  # housing
                         "inconv",  # finance
                         "problematic", "nonprob", "slightly_prob",  # social
                         "not_recom", "priority", "recommended"],  # health
                        [1, 2, 3,  # parents
                         1, 2, 3, 4,  # has_nurs
                         1, 2, 3, 4,  # form
                         4,  # children
                         1, 2, 3,  # housing
                         2,  # finance
                         1, 2, 3,  # social
                         1, 2, 3])  # health

    print(new_df)

    # plot graph
    sns.countplot(x="app_status", data=data)
    plt.show()

    # change the file to anything if needed :)
    new_df.to_csv("data/ICDS_Numeric_Dataset.csv", index=False)


preprocessing("data/train_data.csv")
