from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def machine_learning(k_n, threads):

    # To create a machine learning
    return KNeighborsClassifier(n_neighbors=k_n, n_jobs=threads)


def split_dataset(data, labels, train_test_s=0.2, train_rand=6, test_s=0.5, test_rand=33):

    # To split up a dataset to training and testing groups [Training = 80%, Validating = 10%, Testing = 10%]
    (train_set, test_set, train_labels_set, test_labels_set) = train_test_split(data, labels, test_size=train_test_s, random_state=train_rand, shuffle=True)

    # To split up testing group to validating and testing group
    (valid, test, valid_l, test_l) = train_test_split(test_set, test_labels_set, test_size=test_s, random_state=test_rand, shuffle=True)

    return (train_set, train_labels_set, valid, valid_l, test, test_l)


def train(machine, data, labels):

    # To teach a model via Training group
    machine.fit(data, labels)

    return machine

def get_predict(machine, data):

    return machine.predict(data)


def get_report_by_validating(machine, valid, valid_l, predict):

    return classification_report(valid_l, predict)


def get_accurancy_by_testing(machine, test, test_l):

    return machine.score(test, test_l)


def get_confusion_matrix(test, predict):

    return confusion_matrix(test, predict)

