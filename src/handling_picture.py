import cv2
from matplotlib import pyplot as plt

# To add paddings to passed image
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


def normalization_of_image(image, height=32, width=32):
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
