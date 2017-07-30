#!/usr/bin/python3

from functools import reduce
from itertools import combinations
from sys import argv

import numpy as np

from parse_matrix import parse_problem


def has_inverse(matrix):
    try:
        matrix.I
        return True
    except np.linalg.LinAlgError:
        return False


class FeasibleBase:
    def __init__(self, A, b, z_coeffs, columns):
        self.A = A
        self.b = b
        self.z_size = len(z_coeffs)
        self.z_coeffs = z_coeffs
        self.b_columns = columns

        self.b_values = [i for i in range(self.z_size) if i in columns]
        self.n_values = [i for i in range(self.z_size) if i not in columns]

        self.B = A[:, self.b_values]
        self.N = A[:, self.n_values]

        self.B_i = self.B.I

        self.c_b = np.matrix([z_coeffs[i] for i in self.b_values])
        self.c_n = np.matrix([z_coeffs[i] for i in self.n_values])

    def z(self):
        z_positive = self.c_b * self.B_i * self.b
        z_negative = -1 * (self.c_b * self.B_i * self.N - self.c_n) * self.x_N()
        z = (z_positive + z_negative)[0, 0]
        return z

    def a_(self, n):
        return self.A[:, n]

    def x_B(self):
        return self.B_i * self.b

    def x_N(self):
        return np.matrix([[0 for i in range(self.N.shape[1])]]).T

    def cbbi(self):
        return self.c_b * self.B_i

    def cbbiaj(self, j):
        aj = self.a_(j)
        cbbi = self.cbbi()
        return cbbi * aj

    def cbbiajcj(self, j):
        return self.cbbiaj(j) - self.z_coeffs[j]

    def is_optimal(self):
        is_optimal = True
        for j in range(0, self.z_size):
            val = -1 * (self.cbbiajcj(j))
            if val > 0:
                is_optimal = False
                break
        return is_optimal

    def is_non_negative(self):  # Do not violate non-negative constraints
        return reduce(lambda total, new: total and new, [b >= 0 for b in self.x_B()])

    def best_entering_column(self):
        unused_z_coeffs = [(i, c) for i, c in enumerate(self.z_coeffs) if i not in self.b_columns]
        # Return the index of the highest unused z coefficient
        return max(unused_z_coeffs, key=lambda tup: tup[1])[0]

    def best_exiting_column(self, entering_column):
        w = self.B_i * self.b
        w = [wi[0, 0] for wi in w]

        v = self.B_i * self.a_(entering_column)

        div = [(wi / vi)[0, 0] for wi, vi in zip(w, v)]

        best_min = min([d for d in div if d is not np.inf and d > 0])
        min_index = div.index(best_min)
        return self.b_values[min_index]

    def answer_string(self):
        zstr = 'z = {}\n'.format(self.z())

        basis_i = 0
        x_vars = []
        for i in range(len(self.z_coeffs)):
            if i in self.b_columns:
                x_vars.append(self.x_B()[basis_i, 0])
                basis_i += 1
            else:
                x_vars.append(0)

        x_vars = ['x{} = {}'.format(i+1, v) for i, v in enumerate(x_vars)]
        return zstr + '\n'.join(x_vars)

    def __str__(self):
        return 'z: {}\nx_B: {}'.format(self.z(), self.x_B())

    def __repr__(self):
        return str(self)


class MatrixSimplex:
    def __init__(self, z_coeffs, A, b):
        self.z_coeffs = z_coeffs
        self.A = A
        self.b = b

    def swap_columns(self, base, columns):
        new_column = base.best_entering_column()
        bad_column = base.best_exiting_column(new_column)
        columns.append(new_column)
        columns.remove(bad_column)
        return columns

    def doSimplex(self, brute_force=False):
        A_shape = self.A.shape
        B_size = A_shape[0]
        total_cols = A_shape[1]
        if brute_force:  # This was something I used early on to test my `is_optimal` and `is_non_negative` functions
            # Find all invertible Bases
            feasible_bases = [FeasibleBase(self.A, self.b, self.z_coeffs, columns) for columns in
                              combinations(range(total_cols), B_size) if has_inverse(self.A[:, columns])]

            feasible_bases = [f for f in feasible_bases if f.is_non_negative()]  # Filter out bad bases
            optimal_bases = [f for f in feasible_bases if f.is_optimal()]  # Filter out suboptimal bases

            assert len(optimal_bases) == 1  # Assert that there is, as expected, 1 possible optimal base
            return optimal_bases[0]
        else:  # This is the actual simplex method
            columns = list(range(total_cols - B_size, total_cols))  # Use the last columns first
            while True:
                try:
                    base = FeasibleBase(self.A, self.b, self.z_coeffs, columns)
                    if base.is_optimal() and base.is_non_negative():
                        return base
                    else:
                        columns = self.swap_columns(base, columns)
                except np.linalg.LinAlgError:
                    columns = self.swap_columns(base, columns)


if __name__ == '__main__':
    if '-h' in argv or len(argv) < 2:
        with open('README.md', 'r') as readme:
            print(readme.read())
            exit(0)

    filename = argv[1]
    A, b, z_coeffs = parse_problem(filename)
    simplex = MatrixSimplex(z_coeffs, A, b)

    print(simplex.doSimplex().answer_string())
