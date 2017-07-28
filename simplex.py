#!/usr/bin/python3

from parse_matrix import matrix_from_file
from itertools import combinations
import numpy as np

class FeasibleBase:
    def __init__(self, A, b, z_coeffs, columns):
        self.A = A
        self.b = b
        self.z_size = len(z_coeffs)
        self.b_values = [i for i in range(self.z_size) if i in columns]
        self.n_values = [i for i in range(self.z_size) if i not in columns]

        self.B = A[:, self.b_values]
        self.N = A[:, self.n_values]

        self.B_i = self.B.I

        self.c_b = [z_coeffs[i] for i in self.b_values]
        self.c_n = [z_coeffs[i] for i in self.n_values]

        one = self.c_b * self.B_i 
        two = one * self.b.T
        three = two - (self.c_b * self.B_i * self.N)*self.c_n

    def a_(self, n):
        return self.N[: n]

    def is_optimal(self):
        o = [self.a_(n) for n in self.n_values]
        return reduce(lambda is_optimal, fits: is_optimal and fits, o)


class MatrixSimplex:
    def __init__(self, z_coeffs, A, b):
        self.z_coeffs = z_coeffs
        self.A = A
        self.b = b

    def doSimplex(self):
        # Find all invertible Bases
        A_shape = self.A.shape
        B_size = A_shape[0]
        total_cols = A_shape[1]
        
        feasible_bases = []
        for columns in combinations(range(total_cols), B_size):
            try:
                self.A[:, columns].I # Make sure I exists
                feasible_bases.append(FeasibleBase(self.A, self.b, self.z_coeffs, columns))
            except np.linalg.LinAlgError:
                pass # I did not exist

        return feasible_B

if __name__ == '__main__':
    augmented_mat = matrix_from_file('p2.txt')
    b = augmented_mat[:, -1]
    mat = augmented_mat[:, :-1]
    print(mat, b)

    z_coeffs = [3, 1, 0, 0, 0, 0]
    simplex = MatrixSimplex(z_coeffs, mat, b)

    print(simplex.doSimplex())
