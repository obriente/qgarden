'''
Weight gen adaptive: generates static weight matrices
for the surface code using the adaptive technique from arxiv:1712.02360

Authors: Stephen Spitz and Thomas O'Brien.
Licensed under the GNU GPL 3.0
'''

from .weight_gen_simple import get_correction_matrix
from .weights_moving_average import weights_moving_average


def run(distance, max_lookback, training_data, many_sets=False):

    def gen_pos_lists(d):

        '''
        Generates the position of the ancilla qubits on a(d+1)x(d+1)
        lattice, following the orientation in Fig.1 of arXiv:1705.07855 .
        Note that such an arrangement does not give any spaces for the
        data qubits within the arrays.
        '''

        Z_pos_list = [(2*n + 1 + m % 2, m) for m in range(0, d+1)
                      for n in range(0, (d-1)//2)]
        X_pos_list = [(2*n + m % 2, m) for m in range(1, d)
                      for n in range(0, (d+1)//2)]

        return Z_pos_list, X_pos_list

    pos_lists = gen_pos_lists(distance)

    correction_matrix_X = get_correction_matrix(
        *pos_lists, x_correction_flag=True, distance=distance)
    correction_matrix_Z = get_correction_matrix(
        *pos_lists, x_correction_flag=False, distance=distance)

    num_anc = distance**2-1

    # Generate weight matrices from training dataset
    weight_matrix_object = \
        weights_moving_average(num_anc, max_lookback,
                               sum([len(x) for x in training_data]))

    if many_sets is False:
        training_data = [training_data]

    for dset in training_data:

        weight_matrix_object.new_syndrome()

        for syndrome in dset:

            # Update weight matrices
            weight_matrix_object.update_syndrome(syndrome)

    weight_matrix, boundary_vec = weight_matrix_object.return_weight_matrix()

    return weight_matrix, boundary_vec,\
        correction_matrix_X, correction_matrix_Z
