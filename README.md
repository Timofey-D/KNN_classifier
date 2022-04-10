# KNN_classifier

# Introduction

The program was implemented 2 students from Sami Shamoon College of Engineering. The project is a part of the computer vision course. 


# Description

The program is a Classifier which allows to recognize handwritten Hebrew letters. The project was implemented via K-Nearest Neighbor that classifies the images from dataset.


# Installation

In order to run the program, it is necessary to install several python packages. 

If you don't have a python engine, then also you have to install python using the following command:

Unix
```
sudo apt-get install python
```

Mac
```
brew install python
```

Windows
```
https://www.python.org/ftp/python/3.10.4/python-3.10.4-amd64.exe
```

After that there is a possibility to install the aformentioned packages for successful work with command:
```
pip install opencv-python
pip install sklearn
pip install pandas
pip install numpy
```

# Examples

If you have completed previos steps succussfuly, then you can run the program.
To run the program, you have to move the home directory of the project and to find the ***run*** file. it is a shell script that allows you to run a program and to see results on terminal.

The command looks like:
```
./run
```
or
```
sudo ./run
```
Output should be such:
```
Path to dataset: /Users/timofeydankevich/Desktop/Workplace/Semester_B/Computer_vision/KNN_classifier/hhd_dataset

Neighbors: 5

              precision    recall  f1-score   support

           0       0.78      0.75      0.77        24
           1       0.67      0.54      0.60        26
          10       0.46      0.69      0.55        16
          11       0.22      0.30      0.26        20
          12       0.72      0.62      0.67        29
          13       0.68      0.79      0.73        19
          14       0.47      0.40      0.43        20
          15       0.60      0.71      0.65        21
          16       0.45      0.71      0.56        14
          17       0.78      0.93      0.85        15
          18       0.50      0.53      0.52        15
          19       0.69      0.60      0.64        15
           2       0.79      0.65      0.71        23
          20       0.67      0.13      0.22        15
          21       1.00      0.57      0.73        21
          22       1.00      0.12      0.22        16
          23       0.59      0.81      0.68        16
          24       0.51      0.87      0.65        23
          25       0.89      0.85      0.87        20
          26       0.71      0.57      0.63        21
           3       0.73      0.73      0.73        11
           4       0.75      0.38      0.50        24
           5       0.37      0.69      0.48        16
           6       0.81      0.59      0.68        22
           7       0.83      0.29      0.43        17
           8       0.79      1.00      0.88        11
           9       0.38      0.71      0.49        17

    accuracy                           0.61       507
   macro avg       0.66      0.61      0.60       507
weighted avg       0.67      0.61      0.60       507

Total accurancy: 66.73%
```

