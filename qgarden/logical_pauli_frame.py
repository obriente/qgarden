'''
logical_pauli_frame: a wrapper for the gardener and weight generation
    that deals with a logical Pauli update.

Written by Thomas O'Brien
'''

from . import gardener
from . import weight_gen_simple as weight_gen
from importlib import reload
import numpy as np
reload(gardener)
reload(weight_gen)


def run(data, distance, logical_dic, max_lookback, num_ancillas,
        weight_matrix, boundary_vec, correction_matrices,
        stab_index_left, stab_index_right,
        continuous_flag=True, deriv_flag=2, tbw_tol=0.1):

    '''
    input:
    @ data: list of individual syndromes from a series of experiments
        Each individual syndrome should take the form of a list of time steps
        containing the ancilla measurements and (potential) final stabilizer
        measurements from each time step.

        the ith ancilla measurement from the nth experiment at time step t
        should be accessed at data[n][0][t][i], the ith final
        stabilizer at dat[n][1][i], and the logical operator performed
        after time step t should be accessed at data[n][2][t]

    @ distance: how large of a surface code.

    @ max_lookback: maximum distance back in time for edges to be calculated.
        This really never needs to be longer than ~2*distance.

    @ px, py, pz, pm: error model parameters detailed in arXiv:1705.07855

    @ x_correction_flag: Whether the final measurement is of the X or Z
        stabilizers.

    @ continuous_flag: flag for whether you want results for each time step t.

    @ deriv_flag: flag for the various types of ancilla measurement schemes
        0: assuming feedback as per arXiv:1703.04136
        1: assuming ancilla reset (as per most surface code papers)
        2: assuming no ancilla reset (as per say,
            superconducting architectures)

    @ tbw_tol: the tolerance on the time boundary weight calculation
        made in the gardener.
    '''

    # Initialize gardener
    gard = gardener.Gardener(correction_matrix=None,
                             num_ancillas=num_ancillas,
                             max_lookback=max_lookback,
                             weight_calculation_method='weight_matrix',
                             weight_matrix=weight_matrix,
                             boundary_vec=boundary_vec,
                             deriv_flag=deriv_flag)

    # Initialize list of results
    result = []

    for experiment in data:

        # In this case we just generate a single binary result
        # for each experiment.

        # Reset the gardener
        gard.reset([0]*num_ancillas)

        # Split up syndrome list and final stabilizers
        syndromes, final_stabilizers, logicals = experiment

        logical_Z = 'Z'
        cm_list = []

        # Loop over syndromes, inserting each into the gardener
        for syndrome, logical in zip(syndromes, logicals):
            gard.update(syndrome)

            logical_Z = logical_dic[logical][logical_Z]
            cm_list.append(correction_matrices[logical_Z])
        cm_list.append(correction_matrices[logical_Z])

        # Extract result from gardener. Note that here we can set
        # continue flag to false, as we will not re-use the data,
        # which saves us from having to store and restore the gardener
        # state.
        if logical_Z == 'Z':
            sil = stab_index_left
            sir = stab_index_right
        elif logical_Z == 'X':
            sil = stab_index_left + num_ancillas//2
            sir = stab_index_right + num_ancillas//2
        else:
            raise ValueError('my logical should be on the X or Z')

        result.append(gard.result(final_stabilizers=final_stabilizers,
                                  stab_index_left=sil,
                                  stab_index_right=sir,
                                  continue_flag=False,
                                  cm_list=cm_list))

    return result