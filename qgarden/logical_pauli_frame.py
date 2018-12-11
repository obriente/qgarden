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


def run(data, frame, max_lookback,
        weight_matrix, boundary_vec, code_layout,
        fstab_as_deriv=False,
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

    num_ancillas = code_layout.get_num_anc()
    # Initialize gardener
    gard = gardener.Gardener(correction_matrix=None,
                             code_layout=code_layout,
                             frame=frame,
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

        # Loop over syndromes, inserting each into the gardener
        for syndrome in syndromes:
            gard.update(syndrome)

        # Calculate the stabilizer indices
        H_par = sum([x == 'H' for x in logicals]) % 2
        if H_par == 0:
            sil = 0
            sir = num_ancillas // 2
        else:
            sil = num_ancills // 2
            sir = num_ancills


        corrections = gard.result(
            final_stabilizers=final_stabilizers,
            stab_index_left=sil,
            stab_index_right=sir,
            continue_flag=False,
            get_corrections=True)

        corrections = sorted(corrections, key=lambda x: x[0])

        parities = {'Z': 0}
        frame.reset(active_frame=0, parities=parities)
        time = 0
        for next_time, a1, a2 in corrections:
            while time < next_time:
                frame.apply_clifford(logicals[time])
                time += 1
            frame.update_from_index(a1,a2)
        result.append(frame.get_parity('Z'))

    return result
