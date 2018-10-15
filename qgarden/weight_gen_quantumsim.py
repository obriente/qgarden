'''
weight_gen_simple: Weight generation, using
the error model of arXiv:1703.04136.

(c) 2017 Thomas O'Brien
Distributed under the GNU GPLv3. See LICENSE.txt or
https://www.gnu.org/licenses/gpl.txt

Written by Tom O'Brien

Please note that this is not exactly the same algorithm
as that used in arXiv:1703.04136v1; it has been revamped to fit
the public release of the gardener.

The purpose of this algorithm is to generate a weight matrix
for the gardener in the 'weight_matrix' weight calculation method.

The weight matrix should look like in the following diagram, where
P0 is the path matrix between vertices at the same timestep, and
P is the path matrix between vertices at different timesteps.
Note that there is an abuse of notation here, where 'P_z' and 'P_x'
are used to refer to different parts of these matrices without any
reference. Hopefully it is still fairly clear what is going on here.

In order to generate this, we use the P=1/(1-A) trick from
arXiv:1703.04136, and replace all the P's below by A's.

The A matrices themselves are calculated by hand from the error model.
If someone wants to edit this file to suit another error model, this should
be all that needs to be done!

    #        t         t-1         t-2 .....
    #      ___________________________________
    #      |P0_z,     |P_z^       P_z^T
    #   t  |      P0_x|     P_x^T       P_x^T
    #      |__________|_______________________
    #      |P_z,      |P0_z,     |P_z^T
    #  t-1 |      P_x |      P0_x|      P_x^T
    #      |          |__________|____________
    #      |P_z       |P_z,      |P0_z,     |
    #  t-2 |      P_x |      P_x |      P0_x|
    #      |          |          |__________|_
    #      |          |          |          |

This weight matrix does not contain weights for connections to the
boundary. These are calculated separately and stored in a boundary vector.

Todo: currently this code assumes a square surface.
'''

import numpy as np
from numpy import linalg as la
from . import split

data_qubit_error_function = split.data_qubit_errors
ancilla_qubit_error_function = split.ancilla_qubit_errors
final_data_qubit_error_function = split.final_readout_data_qubit_errors


def run(t1, t2, t_cycle, pm, symm_flag, distance,
        max_lookback, *, two_column_flag=False):
    '''
    Calculate probabilities for all different edges in our matrix,
    convert to weights. Note that in combining probabilities here
    we use a regular sum rather than the slightly-more-accurate
    p1+p2-2*p1*p2, but the difference is negligible.
    Input:

    px,py,pz,pm: error rates

    distance: code distance (must be an odd integer >=3)

    max_lookback: number of timesteps back in time to generate.
    '''

    if distance != 3:
        raise ValueError('Unfortunately, with quantumsim we can only' +
                         ' simulate Surface-17, and not higher distance' +
                         ' codes. As such, this method only works for d=3' +
                         ' (for now)')

    if symm_flag == False:
        raise ValueError('Sorry, I havent put the symmetrization into' +
                         ' this release yet. If it makes you feel better' +
                         ' after a few months work we concluded it does' +
                         ' nothing :(')

    # Generate position lists
    Z_pos_list, X_pos_list, Z_name_list, X_name_list, data_name_list =\
        gen_pos_lists(d=distance)

    # Generate circuit to pass around to different lists
    circuit = split.make_circuit(t1=t1, t2=t2, t_cycle=t_cycle)

    # Get the separated matrices from the model used
    matrix_dic = get_separated_matrices(circuit, pm, distance,
                                        Z_pos_list, X_pos_list,
                                        Z_name_list, X_name_list,
                                        data_name_list)

    # Combine separated matrices into the final product
    weight_matrix, boundary_vec = combine_matrices(matrix_dic, max_lookback,
                                                   two_column_flag)

    # Generate weight matrices from graph error rates and return
    return weight_matrix, boundary_vec


def gen_pos_lists(d):
    '''
    Generates the position of the ancilla qubits on a(d+1)x(d+1)
    lattice, following the orientation in Fig.1 of arXiv:1705.07855 .
    Note that such an arrangement does not give any spaces for the
    data qubits within the arrays.
    '''

    Z_pos_list = [(m, n*2 + (m-1) % 2) for m in range(1, d)
                  for n in range(0, (d+1)//2)]
    X_pos_list = [(m, n+2 - m % 2)for n in range(0, (d-1)//2)
                  for m in range(0, d+1)]
    Z_name_list = ["Z"+str(n) for n in range((d**2-1)//2)]
    X_name_list = ["X"+str(n) for n in range((d**2-1)//2)]
    data_name_list = ["D"+str(n) for n in range(d**2)]

    return Z_pos_list, X_pos_list, Z_name_list, X_name_list, data_name_list


def get_separated_matrices(circuit, pm, distance,
                           Z_pos_list, X_pos_list,
                           Z_name_list, X_name_list,
                           data_name_list):
    '''
    For the sake of readability, I have split up individual pieces of
    weight matrix generation into separate functions. This is an umbrella
    that just returns all of them.
    '''

    A_mat_Z = get_A_mat_Z(circuit, pm, Z_pos_list, Z_name_list, data_name_list)

    A0_mat_Z = get_A0_mat_Z(circuit, distance, Z_pos_list, Z_name_list,
                            X_name_list, data_name_list)

    A_bound_mat_Z = get_A_bound_mat_Z(circuit, distance, Z_pos_list,
                                      Z_name_list, X_name_list, data_name_list)

    A_mat_X = get_A_mat_X(circuit, pm, X_pos_list, X_name_list, data_name_list)

    A0_mat_X = get_A0_mat_X(circuit, distance, X_pos_list, X_name_list,
                            Z_name_list, data_name_list)

    A_bound_mat_X = get_A_bound_mat_X(circuit, distance, X_pos_list,
                                      X_name_list, Z_name_list, data_name_list)

    matrix_dic = {
        'A_mat_Z': A_mat_Z,
        'A0_mat_Z': A0_mat_Z,
        'A_bound_mat_Z': A_bound_mat_Z,
        'A_mat_X': A_mat_X,
        'A0_mat_X': A0_mat_X,
        'A_bound_mat_X': A_bound_mat_X
    }

    return matrix_dic


def get_A_mat_Z(circuit, pm, Z_pos_list, Z_name_list, data_name_list):
    '''
    This function generates the adjacency matrix for the
    Z error graph between different time-steps, and is error
    model specific.
    '''

    # Get number of Z ancillas
    nZ = len(Z_name_list)

    # Initialize matrix. Our error model stretches
    # back in time two steps, so we need twice as many
    # ancillas for the columns
    A_mat_Z = np.zeros([2*nZ, nZ])

    # Vertical errors - loop over Z ancillas
    for label, j1 in zip(Z_name_list, range(nZ)):
        # Measurement error rates are inserted separately
        # and need no calculation
        A_mat_Z[j1+nZ, j1] = pm

        # Perform experiments in quantumsim to calculate ancilla
        # error rates
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        # Extract the error rate we care about from the dictionary
        # returned by quantumsim.
        d_error = (aq_error_data['self_errors'][0] +
                   aq_error_data['self_errors'][1])/2

        A_mat_Z[j1, j1] = d_error

    # Diagonal errors come from data qubits and as such
    # must be generated from their function.
    for label in data_name_list:

        # Perform experiments in quantumsim to calculate data qubit
        # error rates
        dq_error_data = data_qubit_error_function(circuit, label)

        # If we have just one error rate in the following list
        # the data qubit is on the Z-ancilla boundary, and does not
        # make any diagonal errors.
        if len(dq_error_data['x_errors']) == 1:
            continue

        # Extract errors from dictionary
        error1, error2 = dq_error_data['x_errors']

        # This is a diagonal error, so it is asymmetric
        # and the order below is important.
        j1 = Z_name_list.index(error1[0])
        j2 = Z_name_list.index(error1[1])
        A_mat_Z[j1, j2] = error2[2]

    return A_mat_Z


def get_A0_mat_Z(circuit, distance, Z_pos_list, Z_name_list,
                 X_name_list, data_name_list):
    '''
    This function generates the adjacency matrix for the
    Z error graph between the same time step, and is error
    model specific.
    '''

    # Get number of X and Z ancillas
    nZ = len(Z_pos_list)

    # Initialize matrix
    A0_mat_Z = np.zeros([nZ, nZ])

    # Horizontal (data qubit) errors

    for label in data_name_list:

        # Perform experiments in quantumsim to calculate data qubit
        # error rates
        dq_error_data = data_qubit_error_function(circuit, label)

        # If we have just one error rate in the following list
        # the data qubit is on the Z-ancilla boundary, and does not
        # make any horizontal errors between qubits.
        if len(dq_error_data['x_errors']) == 1:
            continue

        # Extract errors from dictionary
        error1, error2 = dq_error_data['x_errors']

        # Figure out which ancillas this error connects and add
        # to weight matrix.
        j1 = Z_name_list.index(error1[0])
        j2 = Z_name_list.index(error1[1])
        A0_mat_Z[j1, j2] = error1[2]
        A0_mat_Z[j2, j1] = error1[2]

    # Hook errors

    for label in X_name_list:
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        for error in aq_error_data['propagated_errors']:

            # If the error only affects a single data qubit,
            # it acts as if it were an error on that data qubit
            if len(error[0]) == 1:

                # Find the data qubit that gets flipped
                dq_label = error[0][0]

                # Find out which ancillas are flipped
                dq_error_data = data_qubit_error_function(circuit, dq_label)

                # If we have just one error rate in the following list
                # the data qubit is on the Z-ancilla boundary, and does not
                # make any horizontal errors between qubits.
                if len(dq_error_data['x_errors']) == 1:
                    continue

                # Extract errors from dictionary
                p_err, _ = dq_error_data['x_errors']

                # Figure out which ancillas this error connects and add
                # to weight matrix.
                j1 = Z_name_list.index(p_err[0])
                j2 = Z_name_list.index(p_err[1])
                A0_mat_Z[j1, j2] += error[1]
                A0_mat_Z[j2, j1] += error[1]

            else:
                # This error propagates from two data qubits onto exactly
                # two ancilla qubits (due to the arrangement of our
                # measurements).

                # Get two data qubits
                dq_label1 = error[0][0]
                dq_label2 = error[0][1]

                # Extract the list of ancillas these generate errors on
                # by re-calling their calibration routines
                dq_error_data1 = data_qubit_error_function(circuit, dq_label1)
                dq_error_data2 = data_qubit_error_function(circuit, dq_label2)
                triggered_ancillas = list(dq_error_data1['x_errors'][0][0:2])
                triggered_ancillas += dq_error_data2['x_errors'][0][0:2]

                if triggered_ancillas[2] == triggered_ancillas[3]:
                    del triggered_ancillas[3]
                if triggered_ancillas[0] == triggered_ancillas[1]:
                    del triggered_ancillas[1]
                # This list probably contains doubled ancillas (i.e.
                # ancillas that are triggered twice), which we have
                # to cancel.
                delete_doubles(triggered_ancillas)

                # There should be exactly two ancillas left, get their
                # indices.
                if len(triggered_ancillas) != 2:
                    raise ValueError('I dont have the right number of ' +
                                     'ancillas triggered here')
                a_label1, a_label2 = triggered_ancillas
                j1 = Z_name_list.index(a_label1)
                j2 = Z_name_list.index(a_label1)

                A0_mat_Z[j1, j2] += error[1]
                A0_mat_Z[j2, j1] += error[1]

    return A0_mat_Z


def get_A_bound_mat_Z(circuit, distance, Z_pos_list, Z_name_list,
                      X_name_list, data_name_list):
    '''
    This function generates the matrix linking the boundary to data
    qubits in the same time-step. Note that we need two separate boundaries
    here for the two edges of the system - each ancilla only ever is matched
    to a single one, and so this is removed when actually doing the blossom
    algorithm.
    '''

    # Standard data qubit errors to the boundary

    # Get number of X and Z ancillas
    nZ = len(Z_pos_list)

    # Connections to the boundary
    A_bound_mat_Z = np.zeros([nZ, 2])

    # Loop over data qubits
    for label in data_name_list:

        # Get list of calibration results
        dq_error_data = data_qubit_error_function(circuit, label)

        # boundary errors occur when there is only one error in the
        # error list; otherwise we don't need to do anything here.
        if len(dq_error_data['x_errors']) > 1:
            continue

        error, = dq_error_data['x_errors']

        # Find out which ancilla this error connects to
        index = Z_name_list.index(error[0])
        pos = Z_pos_list[index]

        # Find out which boundary we connect to
        x1, y1 = pos
        if x1 == 1:
            ci = 0
        else:
            ci = 1

        A_bound_mat_Z[index, ci] += error[2]

    # Ancilla errors can also propagate onto data qubits and cause
    # boundary errors.

    for label in X_name_list:
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        for error in aq_error_data['propagated_errors']:

            # If the error only affects a single data qubit,
            # it acts as if it were an error on that data qubit
            if len(error[0]) == 1:

                # Find the data qubit that gets flipped
                dq_label = error[0][0]

                # Find out which ancillas are flipped
                dq_error_data = data_qubit_error_function(circuit, dq_label)

                # boundary errors occur when there is only one error in the
                # error list; otherwise we don't need to do anything here.
                if len(dq_error_data['x_errors']) > 1:
                    continue

                p_err, = dq_error_data['x_errors']

                # Find out which ancilla this error connects to
                index = Z_name_list.index(p_err[0])
                pos = Z_pos_list[index]

                # Find out which boundary we connect to
                x1, y1 = pos
                if x1 == 1:
                    ci = 0
                else:
                    ci = 1

                A_bound_mat_Z[index, ci] += error[1]

    return(A_bound_mat_Z)


def get_A_mat_X(circuit, pm, X_pos_list, X_name_list, data_name_list):
    '''
    This function generates the adjacency matrix for the
    X error graph between different time steps, and is error
    model specific.
    '''

    # Get number of X ancillas
    nX = len(X_pos_list)

    # Initialize matrix
    A_mat_X = np.zeros([2*nX, nX])

    # Vertical errors - loop over X ancillas
    for label, j1 in zip(X_name_list, range(nX)):
        # Measurement error rates are inserted separately
        # and need no calculation
        A_mat_X[j1+nX, j1] = pm

        # Perform experiments in quantumsim to calculate ancilla
        # error rates
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        # Extract the error rate we care about from the dictionary
        # returned by quantumsim.
        d_error = (aq_error_data['self_errors'][0] +
                   aq_error_data['self_errors'][1])/2

        A_mat_X[j1, j1] = d_error

    # Diagonal errors come from data qubits and as such
    # must be generated from their function.
    for label in data_name_list:

        # Perform experiments in quantumsim to calculate data qubit
        # error rates
        dq_error_data = data_qubit_error_function(circuit, label)

        # If we have just one error rate in the following list
        # the data qubit is on the X-ancilla boundary, and does not
        # make any diagonal errors.
        if len(dq_error_data['z_errors']) == 1:
            continue

        # Extract errors from dictionary
        error1, error2 = dq_error_data['z_errors']
        j1 = X_name_list.index(error1[0])
        j2 = X_name_list.index(error1[1])

        # This is a diagonal error, so it is asymmetric
        # and the order below is important.
        A_mat_X[j1, j2] = error2[2]

    return(A_mat_X)


def get_A0_mat_X(circuit, distance, X_pos_list, X_name_list,
                 Z_name_list, data_name_list):
    '''
    This function generates the adjacency matrix for the
    X error graph at the same time step, and is error
    model specific.
    '''

    # Get number of X ancillas
    nX = len(X_pos_list)

    # Same time steps
    A0_mat_X = np.zeros([nX, nX])

    # Horizontal (data qubit) errors

    for label in data_name_list:

        # Perform experiments in quantumsim to calculate data qubit
        # error rates
        dq_error_data = data_qubit_error_function(circuit, label)

        # If we have just one error rate in the following list
        # the data qubit is on the Z-ancilla boundary, and does not
        # make any horizontal errors between qubits.
        if len(dq_error_data['z_errors']) == 1:
            continue

        # Extract errors from dictionary
        error1, error2 = dq_error_data['z_errors']

        # Figure out which ancillas this error connects and add
        # to weight matrix.
        j1 = X_name_list.index(error1[0])
        j2 = X_name_list.index(error1[1])
        A0_mat_X[j1, j2] = error1[2]
        A0_mat_X[j2, j1] = error1[2]

    # Hook errors

    for label in Z_name_list:
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        for error in aq_error_data['propagated_errors']:

            # If the error only affects a single data qubit,
            # it acts as if it were an error on that data qubit
            if len(error[0]) == 1:

                # Find the data qubit that gets flipped
                dq_label = error[0][0]

                # Find out which ancillas are flipped
                dq_error_data = data_qubit_error_function(circuit, dq_label)

                # If we have just one error rate in the following list
                # the data qubit is on the Z-ancilla boundary, and does not
                # make any horizontal errors between qubits.
                if len(dq_error_data['z_errors']) == 1:
                    continue

                # Extract errors from dictionary
                p_err, _ = dq_error_data['z_errors']

                # Figure out which ancillas this error connects and add
                # to weight matrix.
                j1 = X_name_list.index(p_err[0])
                j2 = X_name_list.index(p_err[1])
                A0_mat_X[j1, j2] += error[1]
                A0_mat_X[j2, j1] += error[1]

            else:
                # This error propagates from two data qubits onto exactly
                # two ancilla qubits (due to the arrangement of our
                # measurements).

                # Get two data qubits
                dq_label1 = error[0][0]
                dq_label2 = error[0][1]

                # Extract the list of ancillas these generate errors on
                # by re-calling their calibration routines
                dq_error_data1 = data_qubit_error_function(circuit, dq_label1)
                dq_error_data2 = data_qubit_error_function(circuit, dq_label2)
                triggered_ancillas = list(dq_error_data1['z_errors'][0][0:2])
                triggered_ancillas += dq_error_data2['z_errors'][0][0:2]
                if triggered_ancillas[2] == triggered_ancillas[3]:
                    del triggered_ancillas[3]
                if triggered_ancillas[0] == triggered_ancillas[1]:
                    del triggered_ancillas[1]

                # This list probably contains doubled ancillas (i.e.
                # ancillas that are triggered twice), which we have
                # to cancel.
                delete_doubles(triggered_ancillas)

                # There should be exactly two ancillas left, get their
                # indices.
                if len(triggered_ancillas) != 2:
                    raise ValueError('I dont have the right number of ' +
                                     'ancillas triggered here')
                a_label1, a_label2 = triggered_ancillas
                j1 = X_name_list.index(a_label1)
                j2 = X_name_list.index(a_label1)

                A0_mat_X[j1, j2] += error[1]
                A0_mat_X[j2, j1] += error[1]

    return A0_mat_X


def get_A_bound_mat_X(circuit, distance, X_pos_list, X_name_list,
                      Z_name_list, data_name_list):
    '''
    This function generates the matrix linking the boundary to data
    qubits in the same time-step. Note that we need two separate boundaries
    here for the two edges of the system - each ancilla only ever is matched
    to a single one, and so this is removed when actually doing the blossom
    algorithm.
    '''

    # Get number of X and Z ancillas
    nX = len(X_pos_list)

    # Connections to the boundary
    A_bound_mat_X = np.zeros([nX, 2])

    # Loop over data qubits
    for label in data_name_list:

        # Get list of calibration results
        dq_error_data = data_qubit_error_function(circuit, label)

        # boundary errors occur when there is only one error in the
        # error list; otherwise we don't need to do anything here.
        if len(dq_error_data['z_errors']) > 1:
            continue

        error, = dq_error_data['z_errors']

        # Find out which ancilla this error connects to
        index = X_name_list.index(error[0])
        pos = X_pos_list[index]

        # Find out which boundary we connect to
        x1, y1 = pos
        if x1 == 1:
            ci = 0
        else:
            ci = 1

        A_bound_mat_X[index, ci] += error[2]

    # Ancilla errors can also propagate onto data qubits and cause
    # boundary errors.

    for label in Z_name_list:
        aq_error_data = ancilla_qubit_error_function(circuit, label)

        for error in aq_error_data['propagated_errors']:

            # If the error only affects a single data qubit,
            # it acts as if it were an error on that data qubit
            if len(error[0]) == 1:

                # Find the data qubit that gets flipped
                dq_label = error[0][0]

                # Find out which ancillas are flipped
                dq_error_data = data_qubit_error_function(circuit, dq_label)

                # boundary errors occur when there is only one error in the
                # error list; otherwise we don't need to do anything here.
                if len(dq_error_data['z_errors']) > 1:
                    continue

                p_err, = dq_error_data['z_errors']

                # Find out which ancilla this error connects to
                index = X_name_list.index(p_err[0])
                pos = X_pos_list[index]

                # Find out which boundary we connect to
                x1, y1 = pos
                if x1 == 1:
                    ci = 0
                else:
                    ci = 1

                A_bound_mat_X[index, ci] += error[1]

    return(A_bound_mat_X)


def combine_matrices(matrix_dic, max_lookback, two_column_flag):

    # Get large matrix dimensions.
    #
    # Our large A matrix looks like:
    #
    #        t         t-1         t-2 .....
    #      ___________________________________
    #      |A0_z      |A_z^T      A_z^T
    #   t  |      A0_x|     A_x^T      A_x^T
    #      |__________|_______________________
    #      |A_z       |A0_z      |A_z^T
    #  t-1 |      A_x |      A0_x|     A_x^T
    #      |          |__________|____________
    #      |A_z       |A_z       |A0_z      |
    #  t-2 |      A_x |      A_x |      A0_x|
    #      |          |          |__________|_
    #      |          |          |          |
    #
    # with the rows and columns stepping back to
    # t-max_lookback.
    # The A_z and A_x matrices must in turn be sliced
    # to fit in with each other in this block diagonal
    # form.

    # For understanding below, note that the width of
    # individual A0_(z,x) and A_(z,x) blocks in the above
    # is n(Z,X), and the width of the boxes above is num_ancillas.

    # Pull matrices from dictionary
    A_mat_Z = matrix_dic['A_mat_Z']
    A0_mat_Z = matrix_dic['A0_mat_Z']
    A_bound_mat_Z = matrix_dic['A_bound_mat_Z']
    A_mat_X = matrix_dic['A_mat_X']
    A0_mat_X = matrix_dic['A0_mat_X']
    A_bound_mat_X = matrix_dic['A_bound_mat_X']

    # Count number of ancillas
    nZ = A0_mat_Z.shape[1]
    nX = A0_mat_X.shape[1]
    num_ancillas = nZ + nX

    # Check the number of time steps that we extend
    # to with single qubit errors.
    # With the current model this will always be 2.
    num_time_steps = A_mat_X.shape[0] // nX

    # Initialize big adjacency matrix
    big_A_mat = np.zeros([num_ancillas*(max_lookback+1),
                          num_ancillas*(max_lookback+1)])

    # Loop over number of previous time-steps needed.
    for j in range(max_lookback+1):

        # Insert matrices for the same time-step, calculating
        # the offsets from the diagram in the function description.
        big_A_mat[j*num_ancillas: j*num_ancillas + nZ,
                  j*num_ancillas: j*num_ancillas + nZ] = A0_mat_Z
        big_A_mat[j*num_ancillas + nZ: (j+1) * num_ancillas,
                  j*num_ancillas + nZ: (j+1) * num_ancillas] = A0_mat_X

        # Run over sections of A_mat
        for k in range(num_time_steps):

            # Make sure we don't overshoot the end of the matrix
            if j + k + 1 > max_lookback:
                break

            # insert Z-ancillas, using the offsets from the diagram
            # in the function description.
            big_A_mat[(j+k+1)*num_ancillas: (j+k+1)*num_ancillas + nZ,
                      j*num_ancillas: j*num_ancillas + nZ] =\
                A_mat_Z[k*nZ:(k+1)*nZ, :]

            big_A_mat[j*num_ancillas: j*num_ancillas + nZ,
                      (j+k+1)*num_ancillas: (j+k+1)*num_ancillas + nZ] =\
                A_mat_Z[k*nZ:(k+1)*nZ, :].transpose()

            # insert X-ancillas, using the offsets from the diagram
            # in the function description.
            big_A_mat[(j+k+1)*num_ancillas + nZ: (j+k+2) * num_ancillas,
                      j*num_ancillas + nZ: (j+1) * num_ancillas] =\
                A_mat_X[k*nX:(k+1)*nX, :]

            big_A_mat[j*num_ancillas + nZ: (j+1) * num_ancillas,
                      (j+k+1)*num_ancillas + nZ: (j+k+2) * num_ancillas] =\
                A_mat_X[k*nX:(k+1)*nX, :].transpose()

    # Use path matrix formula to calculate path probabilities.
    P_mat_big = la.inv(np.identity(num_ancillas*(max_lookback+1))-big_A_mat)

    # We only need a column num_ancillas wide from P_mat_big.
    # Furthermore, we want to take the log of everything to convert
    # to a weight matrix.

    # Numpy log throws up a warning when you take the log of 0,
    # but we actually *want* it to return -inf, so we suppress
    # the warning here
    np.seterr(divide='ignore')
    #weight_matrix = -np.log(P_mat_big[:, :num_ancillas])
    weight_matrix = -np.log(P_mat_big)

    # We now need to calculate the connections from all ancillas
    # to the boundary. As paths never return from the boundary,
    # we can sum over all paths that travel anywhere in the bulk
    # of the code and then make a single step to the boundary.
    boundary_A_mat = np.zeros([num_ancillas, 2])
    boundary_A_mat[:nZ, :] = A_bound_mat_Z
    boundary_A_mat[nZ:, :] = A_bound_mat_X

    boundary_P_mat = P_mat_big[:num_ancillas,
                               :num_ancillas].dot(boundary_A_mat)

    boundary_vec = -np.log(boundary_P_mat[:num_ancillas, :])
    if two_column_flag is False:
        boundary_vec = [min(v) for v in boundary_vec]

    return weight_matrix, boundary_vec


def delete_doubles(a):
    '''
    Finds pairs of objects in a list a and deletes them
    '''
    for x, j in zip(a[::-1], range(len(a)-1, 0, -1)):
        try:
            j2 = a[:j].index(x)
            del a[j]
            del a[j2]
        except:
            pass
