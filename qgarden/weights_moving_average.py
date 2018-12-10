'''
weights_moving average: Implements the adaptive decoder
technique described in arxiv:1712.02360

Author: Stephen Spitz, minor adjustments by Tom O'Brien
and Boris Varbanov.
Licensed under the GNU GPL 3.0
'''
from math import sqrt, log
import numpy as np



class weights_moving_average(object):

    def __init__(self, lookback, window, max_dist, code_layout, *, plotting=False):
        self.code_layout = code_layout
        self.num_anc = self.code_layout.get_num_anc()
        self.lookback = lookback
        self.window = window
        self.max_dist = max_dist
        self.plotting = plotting


        self.measurement_matrix = np.zeros(shape=(2, self.num_anc))
        self.syndrome_matrices = [np.array([])]

        self.xor_matrix = np.zeros(
            shape=(lookback, self.num_anc, self.num_anc))
        self.and_matrix = np.zeros(
            shape=(lookback, self.num_anc, self.num_anc))

        self.var_matrix = np.zeros(
            shape=(lookback, self.num_anc, self.num_anc))
        self.qmat = np.zeros(shape=(lookback, self.num_anc, self.num_anc))
        self.boundary_q = np.zeros(shape=(self.num_anc))
        self.num_measurements = 0

    def new_syndrome(self):
        self.syndrome_matrices.append(np.array([]))
        self.num_measurements = 0

    def update_syndrome(self, new_measurement):

        # Calculate Derivative of Measurements
        new_syndrome = np.logical_xor(
            new_measurement, self.measurement_matrix[1]).astype(int)
        self.num_measurements += 1

        # Pop odd measurement, add new one
        self.measurement_matrix = np.vstack(
            [new_measurement, np.delete(self.measurement_matrix, 1, 0)])

        if self.num_measurements <= 2:
            return

        if len(self.syndrome_matrices[-1]) == 0:
            self.syndrome_matrices[-1] = np.array([new_syndrome])

        elif len(self.syndrome_matrices[-1]) < self.window:
            self.syndrome_matrices[-1] = np.vstack(
                [new_syndrome, self.syndrome_matrices[-1]])

        else:
            self.syndrome_matrices[-1] = np.vstack(
                [new_syndrome, np.delete(self.syndrome_matrices[-1],
                                         self.window - 1, 0)])

    def update_xor_matrix(self):

        self.xor_matrix = np.zeros(shape=self.xor_matrix.shape)
        num_terms = [sum([len(x)-t for x in self.syndrome_matrices])
                     for t in range(self.lookback)]

        for syndrome_matrix in self.syndrome_matrices:
            dim = syndrome_matrix.shape[0]
            if dim == 0:
                continue

            for t in range(self.lookback):
                for i in range(self.num_anc):
                    for j in range(self.num_anc):
                        if t == 0 and i == j:
                            pass
                        else:
                            self.xor_matrix[t, i, j] += sum(np.logical_xor(
                                syndrome_matrix[:dim-t, i],
                                syndrome_matrix[t:, j]).astype(int)) /\
                                num_terms[t]

        if self.plotting:
            plt.figure(figsize=(10,10))
            plt.imshow(np.log(self.xor_matrix[0]))
            plt.colorbar()
            plt.title('log(xor_matrix)')

    def update_and_matrix(self):

        self.and_matrix = np.zeros(shape=self.and_matrix.shape)
        num_terms = [sum([len(x)-t for x in self.syndrome_matrices])
                     for t in range(self.lookback)]
        for syndrome_matrix in self.syndrome_matrices:
            dim = syndrome_matrix.shape[0]
            if dim == 0:
                continue

            for t in range(self.lookback):
                for i in range(self.num_anc):
                    for j in range(self.num_anc):

                        self.and_matrix[t, i, j] += sum(np.logical_and(
                            syndrome_matrix[:dim-t, i],
                            syndrome_matrix[t:, j]).astype(int)) /\
                            num_terms[t]

        if self.plotting:
            plt.figure(figsize=(10,10))
            plt.imshow(np.log(self.and_matrix[0]))
            plt.colorbar()
            plt.title('log(and_matrix)')

    def update_varmat(self):

        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    if (1 - 2 * self.xor_matrix[t, i, j]) == 0:
                        self.var_matrix[t][i][j] = 0
                    else:
                        self.var_matrix[t][i][j] = (
                            self.and_matrix[t, i, j] -
                            self.and_matrix[0, i, i] *
                            self.and_matrix[0, j, j]) /\
                            (1 - 2 * self.xor_matrix[t, i, j])
                        
    def sig_test(self, t, i, j):
        anc_dist = self.code_layout.get_chebyshev_dist(i, j)
        if anc_dist is None or ((anc_dist + abs(t)) > self.max_dist):
            return 0
        return 1

    def update_qmat(self):

        self.update_xor_matrix()
        self.update_and_matrix()
        self.update_varmat()

        for t in range(self.lookback):
            for i in range(self.num_anc):
                for j in range(self.num_anc):
                    Q = 1 - 4 * self.var_matrix[t][i][j]
                    if i == j and t == 0:
                        self.qmat[t][i][j] = 0
                    elif Q < 0:
                        self.qmat[t][i][j] = 0
                    elif 1 - sqrt(Q) < 0:
                        self.qmat[t][i][j] = 0
                    else:
                        self.qmat[t][i][j] = self.sig_test(
                            t, i, j) * (1 - sqrt(Q)) / 2

        if self.plotting:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.log(self.qmat[0]))
            plt.colorbar()
            plt.title('log(qmat)')

        self.update_boundary_vec()

    def return_weight_matrix(self, test_data_flag=False):

        weight_matrix = np.zeros(shape=(self.num_anc*self.lookback,
                                        self.num_anc*self.lookback))
        boundary_weights = np.zeros(shape=(self.num_anc))

        self.update_qmat()

        for n in range(self.num_anc*self.lookback):
            for m in range(n, self.num_anc*self.lookback):
                t = m // self.num_anc - n // self.num_anc
                i = n - (n // self.num_anc) * self.num_anc
                j = m - (m // self.num_anc) * self.num_anc

                if t == 0 and i == j:
                    weight_matrix[n][m] = 0
                else:
                    weight_matrix[n][m] = self.qmat[t][i][j]

        weight_matrix = weight_matrix + np.transpose(weight_matrix)

        weight_matrix = np.linalg.inv(np.identity(self.num_anc*self.lookback) -
                                      weight_matrix)

        exact_boundary_q_left = [self.boundary_q[0]] + [0]*(self.num_anc - 1)
        exact_boundary_q_right = [0]*(self.num_anc - 1) + [self.boundary_q[-1]]

        A_bound_vec = np.transpose(
            np.vstack([np.array(exact_boundary_q_left*self.lookback),
                       np.array(exact_boundary_q_right*self.lookback)]))

        boundary_weights = -np.log(
            np.max(np.dot(weight_matrix, A_bound_vec), axis=1)[:self.num_anc])

        for n in range(self.num_anc*self.lookback):
            for m in range(self.num_anc*self.lookback):

                if weight_matrix[n][m] <= 0:
                    weight_matrix[n][m] = 10000
                elif weight_matrix[n][m] >= 1:
                    weight_matrix[n][m] = .001
                else:
                    weight_matrix[n][m] = -log(weight_matrix[n][m])
        if test_data_flag:
            return self.xor_matrix, self.and_matrix, self.var_matrix, weight_matrix, boundary_weights
        return weight_matrix, boundary_weights

    def update_boundary_vec(self):

        freq = np.array([
            self.and_matrix[0, i, i] for i in range(self.num_anc)])

        self.boundary_q = (
            1 - (1 - 2*freq) * np.prod(1 - 2*self.qmat[0], axis=1) /
            np.multiply(np.prod(np.prod(1 - 2*self.qmat, axis=0), axis=0),
                        np.prod(np.prod(1 - 2*self.qmat, axis=0), axis=1)))/2

        boundary_filter = np.array([self.code_layout.check_boundary(i)
                                    for i in range(self.num_anc)])

        self.boundary_q = self.boundary_q * boundary_filter
