import sys
import os
import random
import numpy as np
import collect_data as collect
import handling_picture as hp
import knn


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


def handle_input(path, _args_):

    # To try getting a 
    try:
        args = sys.argv

        for arg in args:
            temp_path = os.path.join(path, arg)

            if os.path.exists(temp_path) and arg == 'hhd_dataset':
                # To get a complete path to dataset
                path = os.path.join(path, arg)

    except:
        raise Exception("There are not any passed arguments!")

    return path


def load(imagePaths, verbose=1):
    data = []
    labels = []

    for (i, imagePath) in enumerate(imagePaths):
        image = hp.open_image(imagePath)
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
        
    for link in imagePaths:
        open = hp.open_image(link)
        norm_img = hp.normalization_of_image(open)

        data.append(norm_img)


    # show an update every `verbose` images
    if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
        print("[INFO] processed {}/{}".format(i + 1, len(imagePaths)))

    # return a tuple of the data and labels
    return (np.array(data), np.array(labels))


def main():

    # Basic path to the current directory
    path_to_dataset = os.getcwd()

    path_to_dataset = handle_input(path_to_dataset, sys.argv)

    # 3 groups
    training = []
    validating = []
    testing = []

    _groups_ = [ [training, 80], [validating, 10], [testing, 10]]
    _info_ = [ ["Training", training], ["Validating", validating], ["Testing", testing] ]

    get_prepared_list_by_groups(path_to_dataset, _groups_)

    print_info_by_groups(_info_)

    #for link_image in validating:
    #    raw_image = hp.open_image(link_image)
    #    image = hp.normalization_of_image(raw_image)
    #    #hp.print_image(image)
    #    knn.extract_color_histogram(image)
    
    (data, labels) =  load(training)
    #(data, labels) =  load(validating)
    #(data, labels) =  load(testing)
    
    #print(data, labels)

    print( len(data), len(labels))

    data = data.reshape((data.shape[0], 1024))

    print(data)

    le = knn.LabelEncoder()
    labels = le.fit_transform(labels)

    #print(labels)

    (trainX, testX, trainY, testY) = knn.train_test_split(data, labels, test_size=0.30, random_state=46)
    
    print("[INFO] evaluating k-NN classifier...")

    neighbors = [1, 3, 5, 7, 9, 11, 13, 15]

    print(
            len(trainX),
            len(testX),
            len(trainY),
            len(testY)
            )

    for neighbor in neighbors:
        model = knn.KNeighborsClassifier(neighbor, n_jobs=3)

        print("Number of neighbors:", neighbor)
        for i in range(50000):
            model.fit(trainX, trainY)


        #(data, label) = load(testing)
        #test = knn.train_test_split(data, labels, test_size=0.27, random_state=46)

        print(knn.classification_report(testY, model.predict(testX), target_names=le.classes_))
        acc = model.score(trainY, trainX)
        print(acc)


"""
    train = []
    for img in training:
        raw_image = hp.open_image(img)
        image = hp.normalization_of_image(raw_image)
        hist = knn.extract_color_histogram(image)
        train.append(hist)

    print(type(train))

    X_train = np.array(train)
    scaler = knn.preprocessing.StandardScaler().fit(X_train)
    #print(scaler)
    #print(scaler.scale_)

    X, y = knn.make_classification(random_state=42)

    #print(X, y)

    print(X_train[0])

    X_train, X_test, y_train, y_test = knn.train_test_split(X_train, random_state=42)


    print(X_train[0])

    pipe = knn.make_pipeline(knn.StandardScaler(), knn.LogisticRegression())
    pipe.fit(X_train, y_train)  # apply scaling on training data
    print(pipe)

    pipe.score(X_test, y_test)

    print(pipe)

    neigh = knn.KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)

    print(neigh.get_params())
    print(neigh.predict(X_test))


    validate = []
    for img in validating:
        raw_image = hp.open_image(img)
        image = hp.normalization_of_image(raw_image)
        hist = knn.extract_color_histogram(image)
        validate.append(hist)

    test = []
    for img in testing:
        raw_image = hp.open_image(img)
        image = hp.normalization_of_image(raw_image)
        hist = knn.extract_color_histogram(image)
        test.append(hist)
"""
        
if __name__ == '__main__':
    main()

