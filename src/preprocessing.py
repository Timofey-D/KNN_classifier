import os
import cv2
import numpy as np


def __add_padding__(image):
    
    (h, w) = image.shape[:2]

    l_side = (h if h > w else w) / 2

    if h > w:
        segment = int(w + l_side)
        image = cv2.copyMakeBorder(image, 0, 0, segment, segment, cv2.BORDER_REPLICATE)
    else:
        segment = int(h + l_side)
        image = cv2.copyMakeBorder(image, segment, segment, 0, 0, cv2.BORDER_REPLICATE)

    return image


def normalization_image(image, height=32, width=32):
    # To add padding
    p_image = __add_padding__(image) 

    # To get an image size
    (h, w) = p_image.shape[:2]
    m_side = min(h, w)

    # To center an image
    c_image = p_image[
            int(h / 2 - m_side / 2) : int(h / 2 + m_side / 2), 
            int(w / 2 - m_side / 2) : int(w / 2 + m_side / 2)
            ]

    # To change an image size
    r_image = cv2.resize(c_image, (height,width), cv2.INTER_AREA)

    return r_image


# To open an image by passed path
def open_image(image_path):

    try:
        # To open an image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    except:
        print("The path:", image_path)
        raise Exception("The file wasn\'t found or the file doesn't exist!")

    return image


# To print out an image
def print_image(image):
    cv2.imshow('image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_normalized_data(dataset):
    data = []
    labels = []

    # To get a data list with normalized images
    for link_image in dataset.filenames:

        # To get a raw image
        r_image = open_image(link_image)

        # To get a normalized image
        image = normalization_image(r_image)

        # To add the normalizaed image to data
        data.append(image)

        # To get a label
        label = link_image.split(os.path.sep)[-2]

        # to add a lable to labels
        labels.append(label)

    return (np.array(data), np.array(labels))

