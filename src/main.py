import sys
import os
import random
import numpy as np
import collect_data as collect
import handling_picture as hp


"""
    1) To get a path where contains necessary directories
    function 1:
    2) Choose a random directory
    3) Open the chosen directory
    function 2:
    4) To define a number of content in the directory
    5) To define the required number of content in directory
    function 3:
    6) To choose a random image from directory
    7) To put an image into particular list (training, validating or testing)
    8) To delete an image from temporary image list
    9) Repeat steps from 6 to 8 till a number of content of the list doesn't equall the required number of content
"""

def get_prepared_list_by_groups(path, groups):

    used_data = []

    for group in groups:
        
        # To get a random tempoprary list
        temp = collect.get_image_collection(path, group[1], _used_data=used_data)

        # To get already used data
        collect.from_list_to_list(temp, used_data)

        # To assign the temporary list to certain group
        collect.from_list_to_list(temp, group[0])


def check_unique(l_list, r_list):
    
    _min = min(len(r_list), len(l_list))

    for ind in range(_min):
        if r_list[ind] in l_list[ind]:
            return False

    return True


def print_info_by_groups(groups):

    for group in groups:

        print("{}: {}".format(group[0], len(group[1])))


def main():

    # Basic path to the current directory
    path_to_dataset = os.getcwd()

    # To try getting a 
    try:
        args = sys.argv

        for arg in args:
            temp_path = os.path.join(path_to_dataset, arg)

            if os.path.exists(temp_path) and arg == 'hhd_dataset':
                # To get a complete path to dataset
                path_to_dataset = os.path.join(path_to_dataset, arg)

    except:
        raise Exception("There are not any passed arguments!")

    # 3 groups
    training = []
    validating = []
    testing = []

    _groups_ = [ [training, 80], [validating, 10], [testing, 10]]
    _info_ = [ ["Training", training], ["Validating", validating], ["Testing", testing] ]

    get_prepared_list_by_groups(path_to_dataset, _groups_)

    print_info_by_groups(_info_)


if __name__ == '__main__':
    main()

