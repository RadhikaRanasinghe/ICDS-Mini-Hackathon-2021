import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_pairplot(dataset_type):
    data = pd.read_csv(f"data/CRAP_{dataset_type}.csv")
    p = sns.pairplot(data, hue='app_status')
    p.savefig(f"plots/pairplot_{dataset_type}.png")


def count_plot(dataset_type):
    data = pd.read_csv(f"data/CRAP_{dataset_type}.csv")
    sns.countplot(x="app_status", data=data)
    plt.show()
