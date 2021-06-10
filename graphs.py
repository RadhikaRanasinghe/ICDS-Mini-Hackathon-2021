import pandas as pd
import seaborn as sns


def plot_dataset(dataset_type):
    data = pd.read_csv(f"data/ICDS_{dataset_type}_Oversampled_Dataset.csv")
    p = sns.pairplot(data, hue='app_status', kind="kde")
    p.savefig(f"plots/pairplot_{dataset_type}.png")
