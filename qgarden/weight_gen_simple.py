'''
weight_gen_simple: A simple version of weight generation, using
the error model of arXiv:1705.07855.

Written by Tom O'Brien

Please note that this is not exactly the same algorithm
as that used in arXiv:1705.07855v1; it has been revamped to fit
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


def run(px, py, pz, pm, distance, max_lookback, *, x_correction_flag=False):
    '''
    Calculate probabilities for all different edges in our matrix,
    convert to weights. Note that in combining probabilities here
    we use a regular sum rather than the slightly-more-accurate
    p1+p2-2*p1*p2, but the difference is negligible.
    Input:

    px,py,pz,pm: error rates

    distance: code distance (must be an odd integer >=3)

    max_lookback: number of timesteps back in time to generate.

    x_corr_flag: flag for whether we correct x or z errors.
    '''

    # Generate position lists
    Z_pos_list, X_pos_list = gen_pos_lists(d=distance)

    # Convert model error rates to graph error rates
    phx, phy, phz, paZ, paX, \
        psx, psy, psz, pdx, \
        pdy, pdz, pm = convert_probabilities(px, py, pz, pm)

    # Get the separated matrices from the model used
    matrix_dic = get_separated_matrices(phx, phy, phz, paZ, paX, psx, psy,
                                        psz, pdx, pdy, pdz, pm, distance,
                                        Z_pos_list, X_pos_list,
                                        x_correction_flag)

    # Extract the correction matrix to return separately
    correction_matrix = matrix_dic['correction_matrix']

    # Combine separated matrices into the final product
    weight_matrix, boundary_vec = combine_matrices(matrix_dic, max_lookback)

    # Generate weight matrices from graph error rates and return
    return weight_matrix, boundary_vec, correction_matrix


def convert_probabilities(px, py, pz, pm):
    # Convert model error rates to graph error rates

    # Horizontal errors: regardless of whether ancillas are on the
    # boundary or otherwise, we have 4 timesteps outside of the coherent
    # phase, during which all data qubits accumulate errors that lead to
    # horizontal edges.
    phx = 4 * px
    phy = 4 * py
    phz = 4 * pz

    # Ancilla errors: excluding the measurement round, which has a
    # different error effect, we have 6 timesteps for ancilla error
    # per cycle. For this purpose, x and y errors are equivalent.

    # Note the upper-case, as we are converting this from an error
    # to an ancilla label
    paZ = 6 * (px + py)  # Error rate on Z-ancilla
    paX = 6 * (pz + py)  # Error rate on X-ancilla

    # Diagonal errors: Depending on the order of operations and qubit
    # position, diagonal weights are either over 1 or 3 timesteps. We
    # will take care of this later.
    pdx = px
    pdy = py
    pdz = pz

    # Hook errors: By 'hook error' here, we mean any error that
    # propagates from an ancilla qubit to data qubits (1 or 2).
    # I know this is a slight abuse of notation, but it was easier
    # to count errors by thinking about it in this way.
    # As again this is dependent on the position of the qubits, we
    # just store a base value here.
    # Also, as I'm using 'h' for horizontal, I will use 's' for hook
    # because this is a totally sane and sensible thing to do.
    # (More seriously, it's the pattern on the surface code)
    psx = px
    psy = py
    psz = pz

    return phx, phy, phz, paZ, paX, psx, psy, psz, pdx, pdy, pdz, pm


def gen_pos_lists(d):

    '''
    Generates the position of the ancilla qubits on a(d+1)x(d+1)
    lattice, following the orientation in Fig.1 of arXiv:1705.07855 .
    Note that such an arrangement does not give any spaces for the
    data qubits within the arrays.
    '''

    Z_pos_list = [(m, n*2 + (m-1) % 2) for m in range(1, d)
                  for n in range(0, (d+1)//2)]
    X_pos_list = [(m, n*2 - m % 2)for n in range(0, (d-1)//2)
                  for m in range(0, d+1)]

    return Z_pos_list, X_pos_list


def get_separated_matrices(phx, phy, phz, paZ, paX, psx, psy, psz, pdx, pdy,
                           pdz, pm, distance, Z_pos_list, X_pos_list,
                           x_correction_flag):

    '''
    For the sake of readability, I have split up individual pieces of
    weight matrix generation into separate functions. This is an umbrella
    that just returns all of them.
    '''

    correction_matrix = get_correction_matrix(Z_pos_list, X_pos_list,
                                              x_correction_flag, distance)

    A_mat_Z = get_A_mat_Z(paZ, pdx, pdy, pm, Z_pos_list)

    A0_mat_Z = get_A0_mat_Z(phx, phy, psx, psy, pdx, pdy, distance, Z_pos_list)

    A_bound_mat_Z = get_A_bound_mat_Z(phx, phy, psx, psy, pdx,
                                      pdy, distance, Z_pos_list)

    A_mat_X = get_A_mat_X(paX, pdy, pdz, pm, X_pos_list)

    A0_mat_X = get_A0_mat_X(phy, phz, psy, psz, pdy, pdz, distance, X_pos_list)

    A_bound_mat_X = get_A_bound_mat_X(phy, phz, psy, psz, pdy,
                                      pdz, distance, X_pos_list)

    matrix_dic = {
        'correction_matrix': correction_matrix,
        'A_mat_Z': A_mat_Z,
        'A0_mat_Z': A0_mat_Z,
        'A_bound_mat_Z': A_bound_mat_Z,
        'A_mat_X': A_mat_X,
        'A0_mat_X': A0_mat_X,
        'A_bound_mat_X': A_bound_mat_X
    }

    return matrix_dic


def get_correction_matrix(Z_pos_list, X_pos_list, x_correction_flag, distance):

    '''
    This function generates a dictionary of whether or not
    any given chain commutes with a logical error on the surface
    code, when the logical is either X on all physical qubits or
    Z on all physical qubits.
    '''

    # Get number of X and Z ancillas
    nZ = len(Z_pos_list)
    nX = len(X_pos_list)

    # Initialize correction dictionary
    num_ancillas = len(Z_pos_list) + len(X_pos_list)
    correction_matrix = np.zeros([num_ancillas+1,
                                  num_ancillas+1], dtype=int)

    # Check which logical we are making parity for
    if x_correction_flag is True:

        # Loop over X ancillas
        for pos, j1 in zip(X_pos_list, range(nX)):
            x1, y1 = pos

            # Insert parity of connection to boundary
            # This requires we calculate which boundary
            # the ancilla qubit would connect to
            if y1 < distance / 2:
                correction_matrix[j1+nZ, -1] = y1 % 2
                correction_matrix[-1, j1+nZ] = y1 % 2
            else:
                correction_matrix[j1+nZ, -1] = (y1 + 1) % 2
                correction_matrix[-1, j1+nZ] = (y1 + 1) % 2

            # Loop over all other Z ancilla qubits
            for pos2, j2 in zip(X_pos_list[j1+1:], range(j1+1, nX)):
                x2, y2 = pos2
                correction_matrix[j1+nZ, j2+nZ] = (y1-y2) % 2
                correction_matrix[j2+nZ, j1+nZ] = (y1-y2) % 2

    else:

        # Loop over Z ancillas
        for pos, j1 in zip(Z_pos_list, range(nZ)):
            x1, y1 = pos

            # Insert parity of connection to boundary
            # This requires we calculate which boundary
            # the ancilla qubit would connect to
            if x1 < distance / 2:
                correction_matrix[j1, -1] = x1 % 2
                correction_matrix[-1, j1] = x1 % 2
            else:
                correction_matrix[j1, -1] = (x1 + 1) % 2
                correction_matrix[-1, j1] = (x1 + 1) % 2

            # Loop over all other ancilla qubits
            for pos2, j2 in zip(Z_pos_list[j1+1:], range(j1+1, nZ)):
                x2, y2 = pos2

                correction_matrix[j1, j2] = (x1-x2) % 2
                correction_matrix[j2, j1] = (x1-x2) % 2

    return correction_matrix


def get_A_mat_Z(paZ, pdx, pdy, pm, Z_pos_list):

    '''
    This function generates the adjacency matrix for the
    Z error graph between different timesteps, and is error
    model specific.
    '''

    # Get number of Z ancillas
    nZ = len(Z_pos_list)

    # Initialize matrix. Our error model stretches
    # back in time two steps, so we need twice as many
    # ancillas for the columns
    A_mat_Z = np.zeros([2*nZ, nZ])

    # Loop over Z ancillas
    for pos, j1 in zip(Z_pos_list, range(nZ)):
        x1, y1 = pos

        # Measurement and ancilla error rates are vertical
        # and can be immediately inserted
        A_mat_Z[j1+nZ, j1] = pm

        # Note the upper-case, as we have previously converted
        # this from an error to an ancilla label
        A_mat_Z[j1, j1] = paZ

        # Loop over all other ancilla qubits for diagonal connections
        for pos2, j2 in zip(Z_pos_list[j1+1:], range(j1+1, nZ)):
            x2, y2 = pos2

            # Ancillas are directly connected by a single data
            # qubit error if they are one step diagonally from
            # each other.
            if abs(x2-x1) == 1 and abs(y2-y1) == 1:

                # Diagonal errors are of different size and in different
                # direction for the four diagonals.

                # The size is determined by whether we are on the north-west
                # or north-east diagonal.
                if x2-x1 == y2-y1:  # north-west

                    # Only errors during a single step in the coherent phase
                    # become diagonal errors in this situation.
                    if x2 > x1:
                        A_mat_Z[j2, j1] = pdx + pdy
                    else:
                        A_mat_Z[j1, j2] = pdx + pdy

                else:  # north-east

                    # The sequence of data qubit measurements is further apart
                    # in this situation, and so all errors during the coherent
                    # phase become diagonal errors.
                    if x2 < x1:
                        A_mat_Z[j2, j1] = 3 * (pdx + pdy)
                    else:
                        A_mat_Z[j1, j2] = 3 * (pdx + pdy)

    return A_mat_Z


def get_A0_mat_Z(phx, phy, psx, psy, pdx, pdy, distance, Z_pos_list):

    '''
    This function generates the adjacency matrix for the
    Z error graph between the same time step, and is error
    model specific.
    '''

    # Get number of X and Z ancillas
    nZ = len(Z_pos_list)

    # Initialize matrix
    A0_mat_Z = np.zeros([nZ, nZ])

    # Loop over Z ancillas
    for pos, j1 in zip(Z_pos_list, range(nZ)):
        x1, y1 = pos

        # Loop over all other ancilla qubits for horizontal connections
        for pos2, j2 in zip(Z_pos_list[j1+1:], range(j1+1, nZ)):
            x2, y2 = pos2

            # Ancillas are directly connected by a single data
            # qubit error if they are one step diagonally from
            # each other.
            if abs(x2-x1) == 1 and abs(y2-y1) == 1:

                # We combine the horizontal and single qubit hook error rates
                # here. Hook and diagonal error rates
                # depend on whether ancillas are connected along the north-west
                # or north-east diagonal.
                if x2-x1 == y2-y1:  # north-west

                    # Ancilla qubits on the boundary only see a hook error from
                    # a single other ancilla, whilst those in the bulk of the
                    # code see two.
                    if x2 == 0 or x1 == 0 or x2 == distance or x1 == distance:
                        p_tempy = phy + psy + 2 * pdy
                        p_tempx = phx + psx + 2 * pdx
                    else:
                        p_tempy = phy + 2*psy + 2 * pdy
                        p_tempx = phx + 2*psx + 2 * pdx

                    # Combine x and y errors
                    A0_mat_Z[j1, j2] = p_tempx + p_tempy
                    A0_mat_Z[j2, j1] = p_tempx + p_tempy

                else:  # north-east

                    # No single-qubit hook errors occur here
                    A0_mat_Z[j1, j2] = phx + phy
                    A0_mat_Z[j2, j1] = phx + phy

            # two-qubit hook errors occur perpendicularly
            # to the logical operator direction (with the correct
            # choice of gate order, that is).
            if abs(y2-y1) == 2 and abs(x2-x1) == 0:

                A0_mat_Z[j1, j2] = psx + psy
                A0_mat_Z[j2, j1] = psx + psy

    return A0_mat_Z


def get_A_bound_mat_Z(phx, phy, psx, psy, pdx, pdy, distance, Z_pos_list):
    '''
    This function generates the matrix linking the boundary to data
    qubits in the same time-step. Note that we need two separate boundaries
    here for the two edges of the system - each ancilla only ever is matched
    to a single one, and so this is removed when actually doing the blossom
    algorithm.
    '''

    # Get number of X and Z ancillas
    nZ = len(Z_pos_list)

    # Connections to the boundary
    A_bound_mat_Z = np.zeros([nZ, 2])

    # Loop over Z ancillas
    for pos, j1 in zip(Z_pos_list, range(nZ)):
        x1, y1 = pos

        # We connect to the boundary only if we are on the edge of the surface.
        if x1 == 1 or x1 == distance-1:

            # The column index is determined by which edge of the surface
            # we are on
            if x1 == 1:
                ci = 0
            else:
                ci = 1

            # If we are on the corner of the surface, we only connect to the
            # boundary via a single data qubit.
            if y1 == 0 or y1 == distance:
                A_bound_mat_Z[j1, ci] = phx + phy + 3*pdx + 3*pdy
            else:
                A_bound_mat_Z[j1, ci] = (2*phx + 2*phy + 6*pdx + 6*pdy +
                                         psx + psy)

    return(A_bound_mat_Z)


def get_A_mat_X(paX, pdy, pdz, pm, X_pos_list):

    '''
    This function generates the adjacency matrix for the
    X error graph between different time steps, and is error
    model specific.
    '''

    # Get number of X ancillas
    nX = len(X_pos_list)

    # Initialize matrix
    A_mat_X = np.zeros([2*nX, nX])

    # Loop over X ancillas
    for pos, j1 in zip(X_pos_list, range(nX)):
        x1, y1 = pos

        # Measurement and ancilla errors occur vertically
        # and can be inserted immediately
        A_mat_X[j1+nX, j1] = pm

        # Note the upper-case, as we have previously converted
        # this from an error to an ancilla label
        A_mat_X[j1, j1] = paX

        # Loop over all other ancilla qubits for
        # horizontal and diagonal connections
        for pos2, j2 in zip(X_pos_list[j1+1:], range(j1+1, nX)):
            x2, y2 = pos2

            # Ancillas are directly connected by a single data
            # qubit error if they are one step diagonally from
            # each other.
            if abs(x2-x1) == 1 and abs(y2-y1) == 1:

                # Diagonal errors are of different size and in different
                # direction for the four diagonals.

                # The size is determined by whether we are on the north-west
                # or north-east diagonal.
                if x1-x2 == y1-y2:  # north-west

                    # Only errors during a single step in the coherent phase
                    # become diagonal errors in this situation.
                    if x2 > x1:
                        A_mat_X[j1, j2] = pdz + pdy
                    else:
                        A_mat_X[j2, j1] = pdz + pdy

                else:  # north-east

                    # The sequence of data qubit measurements is further apart
                    # in this situation, and so all errors during the coherent
                    # phase become diagonal errors.
                    if x2 < x1:
                        A_mat_X[j2, j1] = 3 * (pdz + pdy)
                    else:
                        A_mat_X[j1, j2] = 3 * (pdz + pdy)

    return(A_mat_X)


def get_A0_mat_X(phy, phz, psy, psz, pdy, pdz, distance, X_pos_list):
    '''
    This function generates the adjacency matrix for the
    X error graph at the same time step, and is error
    model specific.
    '''

    # Get number of X ancillas
    nX = len(X_pos_list)

    # Same time steps
    A0_mat_X = np.zeros([nX, nX])

    # Loop over X ancillas
    for pos, j1 in zip(X_pos_list, range(nX)):
        x1, y1 = pos

        # Loop over all other ancilla qubits for
        # horizontal and diagonal connections
        for pos2, j2 in zip(X_pos_list[j1+1:], range(j1+1, nX)):
            x2, y2 = pos2

            # Ancillas are directly connected by a single data
            # qubit error if they are one step diagonally from
            # each other.
            if abs(x2-x1) == 1 and abs(y2-y1) == 1:

                # We combine the horizontal and single qubit hook error rates
                # Hook and diagonal error rates
                # depend on whether ancillas are connected along the north-west
                # or north-east diagonal.
                if x1-x2 == y1-y2:  # north-west

                    # Ancilla qubits on the boundary only see a hook error from
                    # a single other ancilla, whilst those in the bulk of the
                    # code see two.
                    if x2 == 0 or x1 == 0 or x2 == distance or x1 == distance:
                        p_tempy = phy + psy + 2 * pdy
                        p_tempz = phz + psz + 2 * pdz
                    else:
                        p_tempy = phy + 2*psy + 2 * pdy
                        p_tempz = phz + 2*psz + 2 * pdz

                    A0_mat_X[j1, j2] = p_tempz + p_tempy
                    A0_mat_X[j2, j1] = p_tempz + p_tempy

                else:  # north-east

                    # No single-qubit hook errors occur here
                    A0_mat_X[j1, j2] = phz + phy
                    A0_mat_X[j2, j1] = phz + phy

            # two-qubit hook errors occur perpendicularly
            # to the logical operator direction (with the correct
            # choice of gate order, that is).
            if abs(x2-x1) == 2 and abs(y2-y1) == 0:
                # Hook error
                A0_mat_X[j1, j2] = psz + psy
                A0_mat_X[j2, j1] = psz + psy

    return(A0_mat_X)


def get_A_bound_mat_X(phy, phz, psy, psz, pdy, pdz, distance, X_pos_list):
    '''
    This function generates the matrix linking the boundary to data
    qubits in the same time-step. Note that we need two separate boundaries
    here for the two edges of the system - each ancilla only ever is matched
    to a single one, and so this is removed when actually doing the blossom
    algorithm.
    '''

    # Get number of Z ancillas
    nX = len(X_pos_list)

    # Boundary errors
    A_bound_mat_X = np.zeros([nX, 2])

    # Loop over X ancillas
    for pos, j1 in zip(X_pos_list, range(nX)):
        x1, y1 = pos

        # Boundary errors
        if y1 == 1 or y1 == distance-1:

            # The column index is determined by which edge of the surface
            # we are on
            if y1 == 1:
                ci = 0
            else:
                ci = 1

            # If we are on the corner of the surface, we only connect to the
            # boundary via a single data qubit.
            if x1 == 0 or x1 == distance:
                A_bound_mat_X[j1, ci] = phz + phy + 3*pdz + 3*pdy
            else:
                A_bound_mat_X[j1, ci] = (2*phz + 2*phy + 6*pdz + 6*pdy +
                                         psz + psy)

    return(A_bound_mat_X)


def combine_matrices(matrix_dic, max_lookback):

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
    weight_matrix = -np.log(P_mat_big[:, :num_ancillas])

    # We now need to calculate the connections from all ancillas
    # to the boundary. As paths never return from the boundary,
    # we can sum over all paths that travel anywhere in the bulk
    # of the code and then make a single step to the boundary.
    boundary_A_mat = np.zeros([num_ancillas, 2])
    boundary_A_mat[:nZ, :] = A_bound_mat_Z
    boundary_A_mat[nZ:, :] = A_bound_mat_X

    boundary_P_mat = P_mat_big[:num_ancillas,
                               :num_ancillas].dot(boundary_A_mat)

    boundary_vec = np.zeros([num_ancillas])
    for j in range(num_ancillas):
        boundary_vec[j] = -np.log(np.max(boundary_P_mat[j, :]))

    return weight_matrix, boundary_vec
