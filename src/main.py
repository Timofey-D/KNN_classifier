import os
import sys

from sklearn.datasets import load_files
import preprocessing as pp
import learning as learn
import pandas as pd


def get_info_from_learning(machine, data, labels, X_test_size=0.2, Y_test_size=0.5, rand_train=None, rand_test=None):

    # To get 3 groups of our dataset [Training, Validating, Testing] [80%:10%:10%]
    (training, train_l, valid, valid_l, test, test_l) = learn.split_dataset(data, labels, train_test_s=X_test_size, test_s=Y_test_size, train_rand=rand_train, test_rand=rand_test)

    # To teach our machine
    learn.train(machine, training, train_l)

    # To get a predict for valid group
    predict = learn.get_predict(machine, valid)
    
    # To get a report by validating data
    report = learn.get_report_by_validating(machine, valid, valid_l, predict)

    # To get a total accurancy by testing data
    accurancy = learn.get_accurancy_by_testing(machine, test, test_l)

    # To get a classification matrix
    confusion_matrix = learn.get_confusion_matrix(valid_l, predict)

    return (report, accurancy, confusion_matrix)


def print_machine_info(machine, report, accurancy):
    
    # To get an information about number of neighbors
    number_of_neighbors = machine.get_params()['n_neighbors']

    # To print out a number of k neighbors
    print("Neighbors:", number_of_neighbors)
    print()

    # To print out the report
    print(report)

    # To print out the total accurancy
    print("Total accurancy: {:.2%}".format(accurancy))


def write_to_csv(confusion_matrix, filename='result.csv'):

    # To get a path to the result.csv
    path_to_csv = os.path.join(os.getcwd(), filename)

    # To create a dataframe with confusion matrix
    df = pd.DataFrame(confusion_matrix)

    # To create a csv file with the confusion matrix
    df.to_csv(path_to_csv)


def main():

    try:
        # To get a path to dataset
        path_to_dataset = os.path.join(os.getcwd(), sys.argv[1])
    except:
        raise Exception("The directory does not exist or the command was entered wrong!")

    print("Path to dataset:", path_to_dataset)
    print()

    # T get a dataset by passed path
    dataset = load_files(path_to_dataset)

    # To get a normalized data and labels for teaching
    (data, labels) = pp.get_normalized_data(dataset)

    # To reshape data
    data = data.reshape((data.shape[0], 32*32))

    # To create a machine learning
    machine = learn.machine_learning(5, 3)
    
    # To get information about result of learning
    (report, accurancy, confusion_matrix) = get_info_from_learning(machine, data, labels, rand_train=6, rand_test=33)
    
    # To create a csv file with confusion matrix
    write_to_csv(confusion_matrix, 'confusion_matrix')

    # To print out a result of machine learning
    print_machine_info(machine, report, accurancy)


if __name__ == '__main__':
    main()

