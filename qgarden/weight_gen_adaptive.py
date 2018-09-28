'''
Weight gen adaptive: generates static weight matrices
for the surface code using the adaptive technique from arxiv:1712.02360

Authors: Stephen Spitz and Thomas O'Brien.
Licensed under the GNU GPL 3.0
'''

from .code_layout import CodeLayout
from .weights_moving_average import weights_moving_average


def run(distance, max_lookback, training_data, max_dist, many_sets=False):

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

    Z_pos_lists, X_pos_list = gen_pos_lists(distance)

    anc_data = []
    for pos in Z_pos_lists:
        anc_data.append(('Z', pos))
    for pos in X_pos_list:
        anc_data.append(('X', pos))

    code_layout = CodeLayout(anc_data)

    num_anc = distance**2-1

    # Generate weight matrices from training dataset
    weight_matrix_object = weights_moving_average(num_anc, max_lookback,
                                                  sum([len(x)
                                                       for x in training_data]),
                                                  max_dist=max_dist,
                                                  code_layout=code_layout)

    if many_sets is False:
        training_data = [training_data]

    for dset in training_data:

        weight_matrix_object.new_syndrome()

        for syndrome in dset:

            # Update weight matrices
            weight_matrix_object.update_syndrome(syndrome)

    weight_matrix, boundary_vec = weight_matrix_object.return_weight_matrix()

    return weight_matrix, boundary_vec, code_layout
