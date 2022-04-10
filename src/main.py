import os
import sys

from sklearn.datasets import load_files
import preprocessing as pp
import learning as learn


def main():

    try:
        # To get a path to dataset
        path_to_dataset = os.path.join(os.getcwd(), sys.argv[1])
    except:
        raise Exception("The directory does not exist or the command was entered wrong!")

    print("Path to dataset:", path_to_dataset)
    print()

    # To get a dataset by passed path
    dataset = load_files(path_to_dataset)

    # To get a normalized data and labels for teaching
    (data, labels) = pp.get_normalized_data(dataset)

    # To reshape data
    data = data.reshape((data.shape[0], 32*32))

    # To get 3 groups of our dataset [Training, Validating, Testing] [80%:10%:10%]
    (training, train_l, valid, valid_l, test, test_l) = learn.split_dataset(data, labels)

    # To create a machine learning
    machine = learn.machine_learning(5, 3)
    
    number_of_neighbors = machine.get_params()['n_neighbors']

    print("Neighbors:", number_of_neighbors)
    print()

    # To teach our machine
    learn.train(machine, training, train_l)
    
    # To get a report by validating data
    report = learn.get_report_by_validating(machine, valid, valid_l)

    # To get a total accurancy by testing data
    accurancy = learn.get_accurancy_by_testing(machine, test, test_l)
    
    # To print out the report
    print(report)

    # To print out the total accurancy
    print("Total accurancy: {:.2%}".format(accurancy))


if __name__ == '__main__':
    main()

