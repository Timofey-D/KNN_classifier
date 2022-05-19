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

After that there is a possibility to install the aforementioned packages for successful work with command:
```
pip install opencv-python
pip install sklearn
pip install pandas
pip install numpy
```

# Examples

If you have completed previos steps succussfully, then you can run the program.
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
The program includes 2 modes of execution:
The mode 1 uses the validation set during training and random states are None.
The mode 2 uses the test set during training and random state 1 is 6 and random state 2 is 33.
Enter the program mode [1, 2]: 2
```

**The program creates automatically the Output directory where the results are saved.**


