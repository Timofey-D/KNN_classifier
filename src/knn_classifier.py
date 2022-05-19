import os
import sys

from mode import Mode
from output import Output


def greeting():
    print("The program includes 2 modes of execution:")
    print("The mode 1 uses the validation set during training and random states are None.")
    print("The mode 2 uses the test set during training and random state 1 is 6 and random state 2 is 33.")


def main():

    greeting()
    mode = int(input("Enter the program mode [1, 2]: "))

    program = Mode(mode, sys.argv[1])
    program.run_mode(5, 3)
    knn = program.get_knn()
    report = knn.get_report()

    output = Output(report, program)
    output.create_report_directory()


if __name__ == '__main__':
    main()

