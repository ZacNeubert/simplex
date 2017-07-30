#!/usr/bin/python3

from sys import argv
import numpy as np

def is_int(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def int_line(line):
    return [int(i) for i in line.split(' ') if is_int(i)]

def matrix_from_lines(lines):
    lines = [int_line(line) for line in lines if '#' not in line]
    return np.matrix(lines)

def parse_problem(f):
    with open(f, 'r') as inf:
        lines = inf.readlines()
        A_lines = lines[1:]
        A = matrix_from_lines(A_lines)
        z_coeffs = int_line(lines[0])
        b = A[:, -1]
        A = A[:, :-1]
        return A, b, z_coeffs
