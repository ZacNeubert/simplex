#!/usr/bin/python3

from sys import argv
import numpy as np

def matrix_from_file(f):
    with open(f, 'r') as inf:
        lines = [[int(i) for i in line if i.isnumeric()] for line in inf]
        print(lines)
    return np.matrix(lines)

if __name__ == '__main__':
    print(matrix_from_file('mat.txt'))
