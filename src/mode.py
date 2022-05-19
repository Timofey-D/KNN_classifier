import pandas as pd
import random as rand

from knn import KNN
from preprocessing import Preprocessing


class Mode:

    r_state_1 = None
    r_state_2 = None

    def __init__(self, mode, datadir):

        self.__check_mode__(mode)

        self.datadir = datadir
        self.mode = mode

        if self.mode == 2:
            # r1=56 r2=27, r1=56, r2=75
            self.r_state_1 = 6
            self.r_state_2 = 33 

        self.__data_preparation__()


    def __check_mode__(self, mode):

        modes = [1, 2]

        if modes.count(mode) == 0:
            raise Exception("The obtained mode does not exist!")


    def get_mode(self):
        return self.mode


    def get_knn(self):
        return self.knn


    def get_r_state_1(self):
        return self.r_state_1


    def get_r_state_2(self):
        return self.r_state_2
    

    def get_mode_info(self):
        if self.mode == 1:
            return "The mode 1 runs the training process with validation set.\nThe mode uses random state is None."
        else:
            return "The final mode 2 runs the training process with training set.\nThe mode uses the random state 1 is 6 and the random state 2 is 33."


    def run_mode(self, k_neighbs, n_ths):


        self.knn = KNN(k_neighbs, n_ths, self.data_1, self.data_2, self.labels_1, self.labels_2)
        self.knn.train_model()

    
    def __data_preparation__(self):

        dataset = Preprocessing(self.datadir, 32, 32)
        dataset.reshape_data(2)

        data = dataset.get_data()
        labels = dataset.get_labels()

        (train, validating, train_l, validating_l) = dataset.split_data(data, labels, 0.2, self.r_state_1)
        (valid, test, valid_l, test_l) = dataset.split_data(validating, validating_l, 0.5, self.r_state_2)
        
        self.data_1 = train
        self.labels_1 = train_l

        if self.mode == 1:
            self.data_2 = valid
            self.labels_2 = valid_l
        else:
            self.data_2 = test
            self.labels_2 = test_l

