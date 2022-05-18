import os
import sys

from preprocessing import Preprocessing
from knn import KNN
import pandas as pd


def write_to_csv(confusion_matrix, filename='result.csv'):

    # To get a path to the result.csv
    path_to_csv = os.path.join(os.getcwd(), filename)

    # To create a dataframe with confusion matrix
    df = pd.DataFrame(confusion_matrix)

    # To create a csv file with the confusion matrix
    df.to_csv(path_to_csv)


def main():

    dataset = Preprocessing(sys.argv[1])
    knn = KNN(5, 3)


if __name__ == '__main__':
    main()

