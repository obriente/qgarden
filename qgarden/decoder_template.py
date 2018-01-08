'''
correction_template: a wrapper for the gardener and weight generation.

Written by Thomas O'Brien

This file should interface with data generated by the error model of
arXiv:1705.07855, and produce a string of corrections for use by
that model. Note that it immediately extends to larger distances.

It also hopefully serves as a template for how to use this class.

Please note that we assume the Z ancilla measurements come before the
X ancilla measurements; this has to be hard-coded in the weight generation,
so it makes no sense to change it here.

todo: Currently we assume a square surface, this could be easily changed.
'''

from . import gardener
from . import weight_gen_simple as weight_gen
from importlib import reload
reload(gardener)
reload(weight_gen)


def run(data, distance, max_lookback, px, py, pz, pm, *,
        x_correction_flag=False, continuous_flag=True,
        deriv_flag=2, tbw_tol=0.1):

    '''
    input:
    @ data: list of individual syndromes from a series of experiments
        Each individual syndrome should take the form of a list of time steps
        containing the ancilla measurements and (potential) final stabilizer
        measurements from each time step.

        I.e, if continuous_flag == True,

        the ith ancilla measurement from the nth experiment at time step t
        should be accessed at syndromes[n][t][0][i], and the ith final
        stabilizer at syndromes[n][t][1][i].

        if continuous_flag == False,

        the ith ancilla measurement from the nth experiment at time step t
        should be accessed at syndromes[n][0][t][i], and the ith final
        stabilizer at syndromes[n][1][i]

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

    # Calculate number of ancilla qubits from the code distance
    nX = (distance**2 - 1) // 2
    nZ = nX
    num_ancillas = nX + nZ

    # Calculate position of final stabilizer measurements
    if x_correction_flag is True:
        stab_index_left = nZ
        stab_index_right = num_ancillas
    else:
        stab_index_left = 0
        stab_index_right = nZ

    # Get weight and correction data from the weight generation function
    weight_matrix, boundary_vec, correction_matrix =\
        weight_gen.run(px=px, py=py, pz=pz, pm=pm, distance=distance,
                       max_lookback=max_lookback,
                       x_correction_flag=x_correction_flag)

    # Initialize gardener
    gard = gardener.Gardener(correction_matrix=correction_matrix,
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

        # If we are getting only output from the final round,
        # as in a real experiment
        if continuous_flag is False:

            # Split up syndrome list and final stabilizers
            syndromes, final_stabilizers = experiment

            # Loop over syndromes, inserting each into the gardener
            for syndrome in syndromes:
                gard.update(syndrome)

            # Extract result from gardener. Note that here we can set
            # continue flag to false, as we will not re-use the data,
            # which saves us from having to store and restore the gardener
            # state.
            result.append(gard.result(final_stabilizers=final_stabilizers,
                                      stab_index_left=stab_index_left,
                                      stab_index_right=stab_index_right,
                                      continue_flag=False))

        else:

            # In this case we generate a binary string for each experiment
            # as if we could read the experiment out and keep it too.
            # I like to think of this as an executive experiment order.
            single_result = []

            # Run over each time-step
            for syndrome, final_stabilizers in experiment:

                # Small fix for current datasets, as the X
                # and the Z ancillas have been switched around.
                temp = list(syndrome[:nZ])
                syndrome[:nZ] = list(syndrome[nZ:])
                syndrome[nZ:] = temp

                # Insert single slice of syndrome into gardener
                gard.update(syndrome)

                # Extract result from gardener. Note that here we will re-use
                # the gardener in the loop, and so we must set continue_flag
                # to true so that the gardener state is saved and restored
                sr = gard.result(final_stabilizers=final_stabilizers,
                                 stab_index_left=stab_index_left,
                                 stab_index_right=stab_index_right,
                                 continue_flag=True)
                single_result.append(sr)

            result.append(single_result)

    return result
