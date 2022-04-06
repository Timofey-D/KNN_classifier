import os
import sys
import random


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


def get_required_number(dir_path, ratio):
    # To get the content of the directory
    items = os.listdir(dir_path)

    # To count a number of the content
    quantity = len(items) - 1

    # To get the required number of the items via ratio
    required_number = int((quantity * (ratio)) / 100)

    return required_number


def get_image_collection(path, ratio, extension='png', _used_data=[]):
    collection = []
    content_path = os.listdir(path)

    for dir in content_path:
        if dir.startswith('.'):
            content_path.remove(dir)

    for r_dir in content_path:
        # Path to directory
        dir_path = os.path.join(path, r_dir)

        # To get the required number of items from directory
        req_num_of_content = get_required_number(dir_path, ratio)
        #print("DIR:", len(os.listdir(dir_path)))
        #print("REQ:", req_num_of_content)

        # Path to the directory that contains the images
        content_dir = os.listdir(dir_path)

        temp = []
        while len(temp) != req_num_of_content:
            # To get a random image
            r_image = get_random_item(content_dir)
            p_image = os.path.join(dir_path, r_image)

            if p_image not in temp and p_image not in _used_data:
                # To add the handled image in list
                #print(temp.count(r_image) == 0, _used_data.count(r_image) == 0)
                temp.append(p_image)

            if content_dir.count(r_image) == 1:
                content_dir.remove(r_image)

        # To add elements from tempopary to collection
        from_list_to_list(temp, collection)
    
    return collection


def get_random_item(_list):
    item = 0

    if len(_list) > 0:
        ind = random.randint(0, len(_list) - 1)
    
        item = _list[ind]

    return item


def from_list_to_list(source, target):

    for item in source:
        target.append(item)
