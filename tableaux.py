#!/usr/bin/python3
from sys import argv

import numpy as np

from parse_matrix import parse_problem


class TableauxSimplex:
    def __init__(self, z_coeffs, A, b):
        # Prepare z row
        z_coeffs = [-1 * z for z in z_coeffs]
        self.tab = np.vstack([A, z_coeffs])

        # Prepare z column
        z_col = np.zeros(self.tab.shape[0])
        z_col[-1] = 1
        z_col = np.matrix(z_col).T

        # Prepare z_col and b
        b = np.insert(b, b.shape[0], [0]).T
        self.tab = np.hstack([self.tab, z_col, b])

        # Set up labels
        self.top_labels = list(range(1, self.tab.shape[1] - 1))  # Labels are 0 through the number of vars, 0 being z
        self.top_labels.append(0)
        self.top_labels.append(max(self.top_labels) + 1)
        self.b_label = self.top_labels[-1]

        self.side_labels = self.top_labels[2:-2]
        self.side_labels.append(0)

    def doSimplex(self):
        while not self.is_optimal():
            #print(self)
            smallest = min(self.z_row)
            smallest_col = self.z_row.index(smallest)
            #print('Pivoting on col', smallest_col)
            self.pivot(smallest_col)
        for i, row in enumerate(self.tab):
            if self.b_col[i] < 0:
                self.tab[i] *= -1
        print(self.as_answer())

    def pivot(self, p_col):
        b_vals = self.b_col / self.col(p_col)
        b_val_list = [b[0] if b[0] >= 0 else np.inf for b in b_vals.tolist()[:-1]]
        p_row = b_val_list.index(min(b_val_list))
        #print('Pivoting on row', p_row)

        # Normalize to 1
        self.tab[p_row] = self.tab[p_row] / self.tab[p_row, p_col]
        assert self.tab[p_row, p_col] == 1

        # Zero out all rows
        for i, row in enumerate(self.tab):
            if i != p_row:
                self.tab[i] -= self.tab[p_row] * self.tab[i, p_col]

        # Swap Labels
        self.side_labels[p_row], self.top_labels[p_col] = self.top_labels[p_col], self.side_labels[p_row]

    @property
    def z_row(self):
        return self.tab[-1].tolist()[0]

    @property
    def b_col(self):
        return self.tab[:, -1]

    def col(self, i):
        return self.tab[:, i]

    def is_optimal(self):
        return min(self.z_row) >= 0

    def str_labels(self, labels):
        return ['z' if l == 0 else 'b' if l == self.b_label else 'x{}'.format(l) for l in labels]

    @property
    def verbose_top_labels(self):
        return self.str_labels(self.top_labels)

    @property
    def verbose_side_labels(self):
        return self.str_labels(self.side_labels)

    @property
    def truncated_tab(self):
        truncated = self.tab.copy()
        for i in range(truncated.shape[0]):
            for j in range(truncated.shape[1]):
                truncated[i, j] = float('{:.2f}'.format(truncated[i, j]))
        return truncated

    def __str__(self):
        tab_side_labels = '\n'.join(
            [' '.join([l, t]) for l, t in zip(self.verbose_side_labels, [str(row) for row in self.tab])]
        )
        return '\n'.join(
            (' ' * 6 + (' ' * 2).join(self.verbose_top_labels),
             tab_side_labels))

    def __repr__(self):
        return str(self)

    def var_values(self):
        vals = [0 for i in range(1, self.b_label)]
        for i, label in enumerate(self.side_labels[:-1]):
            vals[label - 1] = self.b_col[i, 0]
        return vals

    def as_answer(self):
        return '\n'.join([str(self),
                          '',
                          'z = {}'.format(self.tab[-1, -1]),
                          *['x{} = {}'.format(i + 1, v) for i, v in enumerate(self.var_values())]])


if __name__ == '__main__':
    if '-h' in argv or len(argv) < 2:
        with open('README.md', 'r') as readme:
            print(readme.read())
            exit(0)

    filename = argv[1]
    A, b, z_coeffs = parse_problem(filename)
    simplex = TableauxSimplex(z_coeffs, A, b)
    simplex.doSimplex()
