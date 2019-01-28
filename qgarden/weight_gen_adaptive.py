'''
Weight gen adaptive: generates static weight matrices
for the surface code using the adaptive technique from arxiv:1712.02360

Authors: Stephen Spitz and Thomas O'Brien.
Licensed under the GNU GPL 3.0
'''

from .weights_moving_average import weights_moving_average


def run(max_lookback, training_data, code_layout, max_dist, many_sets=False, plotting=False):

    if many_sets is False:
        training_data = [training_data]

    # Generate weight matrices from training dataset
    weight_matrix_object = weights_moving_average(max_lookback,
                                                  sum([len(x)
                                                       for x in training_data]),
                                                  max_dist,
                                                  code_layout,
                                                  plotting=plotting)

    for dset in training_data:

        weight_matrix_object.new_syndrome()

        for syndrome in dset:

            # Update weight matrices
            weight_matrix_object.update_syndrome(syndrome)

    weight_matrix, boundary_vec = weight_matrix_object.return_weight_matrix()

    return weight_matrix, boundary_vec
