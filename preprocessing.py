import pandas as pd


def preprocessing(dataset):
    file_handler = open(dataset, "r")
    data = pd.read_csv(file_handler, sep=",")
    df = pd.DataFrame(data)  # define dataframe
    file_handler.close()

    print(data)

    # replace the strings with integers in dataframe
    new_df = df.replace(["usual", "pretentious", "great_pret",  # parents
                         "less_proper", "improper", "proper", "critical", "very_crit",  # has_nurs
                         "foster", "incomplete", "complete", "completed",  # form
                         "more",  # children
                         "less_conv", "convenient", "critical",   # housing
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

    print(new_df)

    # change the file to anything if needed :)
    new_df.to_csv("data/ICDS_Numeric_Dataset.csv", index=False)


preprocessing("data/train_data.csv")
