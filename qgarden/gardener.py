'''Gardener: a wrapper for a blossom algorithm.

Written by Tom O'Brien, with much consultation from Brian Tarasinski.

In order to do accurate error correction for a quantum computer
via the blossom algorithm, it is required that we extract errors
from a syndrome measurement, convert them to vertices on an error
graph, and calculate weights between nearby vertices on the graph
(to some degree of accuracy). These vertices and weights can then
be fed into the blossom algorithm itself. After the algorithm
finishes, it is also necessary to extract the final pairing, calculate
a parity from this, and return this to the user.

This wrapper intends to be a handle to do all of the above, in as
many different ways as possible. This includes pre-calculating
weight matrices from initial error measurements (or data from an
error model), or taking input weight matrices, or doing on-the-fly
weight calculation as described in arXiv:1703.04136.

This wrapper interfaces directly with a home-made blossom algorithm,
designed to work with the special requirements of QEC codes. It is
not based on Kolmogorov's Blossom V, which is a far superior piece
of code written by someone that actually knew what they were doing.
Importantly, it should be noted that since this was written we found
many small tricks to decrease the overhead. Maybe one day these will
find there way into here as well.

The input into the gardener comes in two forms; the made measurements
and the weight matrices used for weight calculations. The indexing
for these is specified here:
    Let N be the number of ancillas.
    At timestep t, weight_matrix[i,j] contains the weight between
    the ancilla i%N measurement at time t-i//N,
    and the ancilla j%N measurement at time t-j//N

TODO:
    include direct calculation of weight matrix terms given
calibration data.
    include correlation calculations a la Fowler/Delfosse.
    include weight matrix re-weighting based on correlations.
'''

import numpy as np
from . import blossom as bl
from math import log
import scipy.sparse as sp
from importlib import reload
reload(bl)


class Gardener:

    def __init__(self, *,
                 code_layout,
                 num_ancillas,
                 frame=None,
                 max_lookback=50,
                 weight_calculation_method='weight_matrix',
                 **kwargs):
        '''
        Upon initialization, we perform any required pre-calculation
        of weight matrices, and store some technical data for later.

        Required input:
        @code_layout: For any given string between two errors, we need
            to know whether or not this commutes with the logical operator,
            which is dependent on the definition of the logical operator.

            (Here, by logical we mean either logical X or logical Z. It
            should always be clear from the situation.)

        @frame: a HeisenbergFrame that keeps track of the parity of a
            set of logical operators which both evolve in time and pick
            up errors.

        @weight_calculation_method: flag for how weights are calculated
            for input into blossom.

            options:

            'weight_matrix': Full weight matrix for ancilla graph, plus
                dictionary for boundary connections.
                requires: 'weight_matrix', 'boundary_vec',
                'boundary_vec_final'

            'partial_weight_matrix': AP0, P0, and Pstart matrices for
                generation of a weight matrix at different timesteps
                conditional on ancilla outcomes, plus dictionary for
                boundary connections.
                requires: 'Pstart_mats', 'P0_mats', 'AP0_mats',
                'P0_mats_final', 'AP0_mats_final', 'boundary_vec',
                'time_boundary_vec', 'boundary_vec_final'

        @num_ancillas: Tells us how many measurements to expect each round.

        Options (kwargs):
        @max_lookback: Cutoff for the weight matrix: tells us how far back
            in time to look for weights.
        @prob_tol: The cutoff for a probability to be too small to
            be fed to blossom as a weight.
        @deriv_flag: A flag to determine whether the input to the
            gardener is:
            0 - ancilla measurements with feedback
            1 - ancilla measurements with reset
            2 - ancilla measurements without reset
        @tbw_tol: A tolerance on the time-boundary weight - essentially
            an additional buffer to prevent our approximation being too
            large.
        @only_final_flag: A special flag for training a machine learning
            decoder (see ArXiv:1705.07855) that spits out only parities
            from errors that connect to the final stabilizer measurements.
        '''

        # Store the weight calculation method
        self.weight_calculation_method = weight_calculation_method

        # In order to interpret the final pairing, we need
        self.code_layout = code_layout

        # Store the number of ancilla measurements per cycle
        self.num_ancillas = num_ancillas

        # Store the number of rows we need to keep in the weight matrix
        self.max_rows = (max_lookback + 1) * num_ancillas
        self.max_lookback = max_lookback

        self.frame=frame

        # Tolerance for the probabilities in the P_matrices
        try:
            self.max_weight = kwargs['max_weight']
        except:
            self.max_weight = 10

        # Determines which derivative of our results we need to take
        try:
            self.deriv_flag = kwargs['deriv_flag']
        except:
            self.deriv_flag = 2

        # Special flag for training NN
        try:
            self.only_final_flag = kwargs['only_final_flag']
        except:
            self.only_final_flag = False

        # Store weight variables depending on what we want to do
        if weight_calculation_method == 'partial_weight_matrix':

            # Determines whether AP0_mats
            # is a single matrix or a list of matrices dependent
            # on the previous ancilla rounds.
            try:
                self.symm_flag = kwargs['symm_flag']
            except:
                self.symm_flag = True

            # Matrices for weight calculations
            self.Pstart_mat = kwargs['Pstart_mat']
            self.P0_mat = kwargs['P0_mat']
            self.AP0_mats = kwargs['AP0_mats']
            self.P0_mat_final = kwargs['P0_mat_final']
            self.AP0_mats_final = kwargs['AP0_mats_final']

            # Dictionaries for boundary weight calculations
            self.boundary_vec = kwargs['boundary_vec']
            self.boundary_vec_final = kwargs['boundary_vec_final']

            # list for quick conversion of a list of 0's and 1's into a number
            self.convert_vec = reversed([2**n
                                         for n in range(self.num_ancillas)])

            # distance between vertical rounds for use in blossom.
            self.time_boundary_weight = -0.5*np.log(
                np.max(self.AP0_mats[num_ancillas:, :])**2 +
                np.max(self.AP0_mats[:num_ancillas, :]))

        elif weight_calculation_method == 'weight_matrix':

            # In this case the weight matrix must be symmetric
            # (i.e. independent of any ancilla readout).
            self.symm_flag = True

            # lookup tables for weight extraction
            self.weight_matrix = kwargs['weight_matrix']

            # We allow the possibility for the user to specify a different
            # weight matrix for the final round
            try:
                self.weight_matrix_final = kwargs['weight_matrix_final']
            except:
                self.weight_matrix_final = kwargs['weight_matrix']

            self.boundary_vec = kwargs['boundary_vec']

            # We allow the possibility for the user to specify a different
            # weight matrix for the final round
            try:
                self.boundary_vec_final = kwargs['boundary_vec_final']
            except:
                self.boundary_vec_final = kwargs['boundary_vec']

            if self.max_rows > self.weight_matrix.shape[0]:
                raise ValueError('Lookback distance too large ' +
                                 'for given weight matrix.')

            # distance between vertical rounds for use in blossom
            tbw = np.min([self.weight_matrix[j + self.num_ancillas*n, j]/n
                          for j in range(self.num_ancillas)
                          for n in range(max_lookback)])
            self.time_boundary_weight = tbw

        else:
            raise ValueError('Weight calculation method not understood')

        # Subtract a tolerance from the time boundary weight
        try:
            self.time_boundary_weight -= kwargs['tbw_tol']
        except:
            self.time_boundary_weight -= 0.1

        # Reset prepares the gardener for input. We perform it here
        # so that the user doesn't have to reset before using this.
        self.reset()

    def update_weights(self, **kwargs):

        # Updates any selection of weight data within the gardener
        # Assumes that we want to keep the same calculation method
        # (otherwise may as well start a new gardener).
        # Also updates the time_boundary_weight within the
        # graph if it becomes smaller.

        # Tolerance for the probabilities in the P_matrices
        try:
            self.max_weight = kwargs['max_weight']
        except:
            self.max_weight = 10

        # Store weight variables depending on what we want to do
        if self.weight_calculation_method == 'partial_weight_matrix':

            # Determines whether AP0_mats
            # is a single matrix or a list of matrices dependent
            # on the previous ancilla rounds.
            try:
                self.symm_flag = kwargs['symm_flag']
            except:
                self.symm_flag = True

            # Matrices for weight calculations
            try:
                self.Pstart_mat = kwargs['Pstart_mat']
            except:
                pass

            try:
                self.P0_mat = kwargs['P0_mat']
            except:
                pass

            try:
                self.AP0_mats = kwargs['AP0_mats']
            except:
                pass

            try:
                self.P0_mat_final = kwargs['P0_mat_final']
            except:
                pass

            try:
                self.AP0_mats_final = kwargs['AP0_mats_final']
            except:
                pass

            # Dictionaries for boundary weight calculations
            try:
                self.boundary_vec = kwargs['boundary_vec']
            except:
                pass

            try:
                self.boundary_vec_final = kwargs['boundary_vec_final']
            except:
                pass

            # list for quick conversion of a list of 0's and 1's into a number
            self.convert_vec = reversed([2**n
                                         for n in range(self.num_ancillas)])

            # distance between vertical rounds for use in blossom.
            self.time_boundary_weight = -0.5*np.log(
                np.max(self.AP0_mats[self.num_ancillas:, :])**2 +
                np.max(self.AP0_mats[:self.num_ancillas, :]))

        elif self.weight_calculation_method == 'weight_matrix':

            # In this case the weight matrix must be symmetric
            # (i.e. independent of any ancilla readout).
            self.symm_flag = True

            # lookup tables for weight extraction
            try:
                self.weight_matrix = kwargs['weight_matrix']
            except:
                pass

            # We allow the possibility for the user to specify a different
            # weight matrix for the final round
            try:
                self.weight_matrix_final = kwargs['weight_matrix_final']
            except:
                try:
                    self.weight_matrix_final = kwargs['weight_matrix']
                except:
                    pass

            try:
                self.boundary_vec = kwargs['boundary_vec']
            except:
                pass

            # We allow the possibility for the user to specify a different
            # weight matrix for the final round
            try:
                self.boundary_vec_final = kwargs['boundary_vec_final']
            except:
                try:
                    self.boundary_vec_final = kwargs['boundary_vec']
                except:
                    pass

            if self.max_rows > self.weight_matrix.shape[0]:
                raise ValueError('Lookback distance too large ' +
                                 'for given weight matrix.')

            # distance between vertical rounds for use in blossom
            tbw = np.min([self.weight_matrix[j + self.num_ancillas*n, j]/n
                          for j in range(self.num_ancillas)
                          for n in range(self.max_lookback)])
            self.time_boundary_weight = tbw

        # Subtract a tolerance from the time boundary weight
        try:
            self.time_boundary_weight -= kwargs['tbw_tol']
        except:
            self.time_boundary_weight -= 0.1

        self.graph.time_boundary_weight = min(self.time_boundary_weight,
                                              self.graph.time_boundary_weight)

    def reset(self, ancilla_states=None):
        '''Remove previous data about a given experimental run if it is present,
        and set up system for a new run.

        Input:

        ancilla_states: a list of length num_ancillas containing the
            state of the ancillas at the start of the experiment

        Output: none
        '''

        if self.deriv_flag == 0:
            self.current_stabilizers = [0]*self.num_ancillas

        # Reset graph
        self.graph = bl.Bloss(self.time_boundary_weight)

        # Reset error list
        self.full_error_list = []
        self.num_errors = 0

        # Reset timestep and counts
        self.timestep = 0
        if ancilla_states is None:
            self.syndromes = [0]*self.num_ancillas*2
        else:
            self.syndromes = ancilla_states * 2

        # Reset weight matrices to empty set if we are creating them
        # on-the-fly
        if self.weight_calculation_method == 'partial_weight_matrix':
            self.P_mat = sp.csc_matrix((self.num_ancillas, self.num_ancillas),
                                       dtype='float')

    def update(self, syndrome):
        '''Store syndrome data, do any immediate correction we want

        Input:
        syndrome: n bits representing the current measurement

        Output: none
        '''

        # Sanitize input
        syndrome = [int(x) for x in syndrome]

        # Update time
        self.timestep += 1

        # Adds one time-step of everything to blossom
        # so that it knows where to halt.
        # Note: as defined, this should be done *before*
        # the new vertices are inserted.
        self.graph.add_t_weight()

        # Interpret syndrome
        if self.deriv_flag == 0:
            self.store_syndrome(syndrome)
            error_list = self.get_new_errors(syndrome)
        else:
            syndromedd = self.update_syndrome(syndrome)
            error_list = self.get_new_errors(syndromedd)

        # Update weight matrices
        # Note that the chance of an error occurring on an ancilla in this time
        # step is dependent on the state of the ancilla in the previous time
        # step
        if self.weight_calculation_method == 'partial_weight_matrix':
            self.update_Pmats(self.syndromes[:self.num_ancillas])

        # Insert errors into blossom.
        for error in error_list:
            weight_list = self.get_weights(error=error,
                                           final_flag=0)
            self.graph.add_vertex(weight_list)

        # Runs blossom till it halts
        self.graph.run()

    def result(self,
               boundary_switch=1,
               continue_flag=False,
               final_stabilizers=None,
               syndromedd=None,
               stab_index_left=None,
               stab_index_right=None):
        '''Finish running blossom and return the most likely correction.

        Input:
        final_stabilizers: the last stabilizer measurements generated
            from explicit readout of the data qubits.
        stab_index_left, stab_index_right: the slice of self.syndromes
            corresponding to the final stabilizers
        boundary_switch: either 0, 1, 2:
            0 - calculates without final stabilizers
            1 - calculates with final stabilizers
            2 - calculates both
        cm1, cm2: optional alternative correction matrices.

        Output:
        res: the logical error bitflip bit.
        '''
        if boundary_switch > 0:

            res = 0

            # Check for any last errors, and add them as necessary to the
            # graph. Determine if any errors in the final round exist
            if self.deriv_flag == 0:
                # This is a special case, where we have been performing
                # feedback to drive ancilla qubits into the ground state,
                # and as such have to store the current stabilizer separately.
                curr_stab = self.current_stabilizers[stab_index_left:
                                                     stab_index_right]

            elif self.deriv_flag == 1:
                # This is the case with ancilla reset, and so the syndromes
                # measure the actual stabilizers.
                curr_stab = self.syndromes[stab_index_left:stab_index_right]

            elif self.deriv_flag == 2:

                # This is the case with no ancilla reset, and so the
                # stabilizers are measured in the difference between the
                # ancilla measurements.
                curr_syn = self.syndromes[stab_index_left:stab_index_right]
                prev_syn = self.syndromes[self.num_ancillas +
                                          stab_index_left:
                                          self.num_ancillas +
                                          stab_index_right]

                curr_stab = [x ^ y for x, y in zip(curr_syn, prev_syn)]

            # Final errors come from the difference between the directly
            # measured stabilizers and the stabilizers measured indirectly
            # in previous rounds.
            if syndromedd is None:
                # Sanitize input
                final_stabilizers = [int(x) for x in final_stabilizers]
                syndromedd = [x ^ y for x, y in zip(final_stabilizers,
                                                    curr_stab)]

            # Use the same functions as in update to insert errors
            error_list = self.get_new_errors_final(syndromedd,
                                                   stab_index_left,
                                                   stab_index_right)
            # If there is nothing to correct, do nothing
            if self.num_errors > 0:

                # Update weight matrices
                # Note that the chance of an error occurring on an ancilla in
                # this time step is dependent on the state of the ancilla in
                # the *previous* time step.
                if self.weight_calculation_method == 'partial_weight_matrix':
                    pmat = self.get_Pmats_final(
                        self.syndromes[:self.num_ancillas])
                else:
                    pmat = None

                # Compound weight lists
                weight_lists = []
                for error in error_list:
                    weight_list = self.get_weights(error=error,
                                                   final_flag=1,
                                                   final_pmat=pmat)
                    weight_lists.append(weight_list)

                # Runs blossom till it halts
                pairing = self.graph.finish(boundary_list=None,
                                            weight_lists=weight_lists,
                                            c_flag=continue_flag)

                for index, pair in zip(range(len(pairing)), pairing):
                    if pair is None:
                        continue

                    if pair == 0:  # Connection to the boundary
                        ancilla_index = (self.full_error_list[index-1][0])
                        pair_index = -1
                        timestep = self.full_error_list[index-1][1]

                    elif pair < index:  # Connection between two ancillas
                        ancilla_index = (self.full_error_list[index-1][0])
                        pair_index = self.full_error_list[pair-1][0]
                        timestep = min(self.full_error_list[pair-1][1],
                                       self.full_error_list[index-1][1])

                    else:
                        continue

                    if self.frame:
                        frame.update_from_index(ancilla_index, pair_index)
                    else:
                        res = res ^ self.code_layout.get_correction(
                            ancilla_index, pair_index, stab_index_left, stab_index_right)

        if boundary_switch != 1:

            res2 = 0

            # Runs blossom till it halts
            pairing = self.graph.finish(boundary_list=None,
                                        weight_lists=[],
                                        c_flag=continue_flag)

            for index, pair in zip(range(len(pairing)), pairing):
                if pair is None:
                    continue

                if pair == 0:  # Connection to the boundary
                    ancilla_index = self.full_error_list[index-1][0]
                    pair_index = -1
                    timestep = self.full_error_list[index-1][1]

                elif pair < index:  # Connection between two ancillas
                    ancilla_index = self.full_error_list[index-1][0]
                    pair_index = self.full_error_list[pair-1][0]
                    timestep = min(self.full_error_list[pair-1][1],
                                   self.full_error_list[index-1][1])

                else:
                    continue
                if self.frame:
                    frame.update_from_index(ancilla_index, pair_index)
                res2 = res2 ^ self.code_layout.get_correction(
                    ancilla_index, pair_index, stab_index_left, stab_index_right)

        # Undo any damage done by final measurement
        if continue_flag and len(error_list) > 0:
            del self.full_error_list[-len(error_list):]
            self.num_errors -= len(error_list)
        if frame:
            return
        elif boundary_switch == 0:
            return res2
        elif boundary_switch == 1:
            return res
        else:
            return res, res2

    def update_syndrome(self, syndrome):
        '''Stores syndrome, takes second derivative, returns

        Input:
        syndrome: n bits representing current measurement

        Output:
        syndromedd: the second time derivative of the syndrome.
        '''
        # Take either first or second derivative depending on deriv_flag
        syndromedd = [x ^ y for x, y in zip(syndrome,
                                            self.syndromes[self.num_ancillas * (2-self.deriv_flag):
                                                           self.num_ancillas * (3-self.deriv_flag)])]

        # Push syndrome onto stack, eject last row as we no longer need it
        self.syndromes = (self.syndromes[self.num_ancillas:] + syndrome)

        return syndromedd

    def store_syndrome(self, syndrome):
        '''Stores syndrome, assuming that a second derivative has already been taken
        Also updates the current guess at the stabilizer measurements.

        Input:
        syndrome: n bits representing current measurement
        '''

        # Push syndrome onto stack, eject last row as we no longer need it
        self.current_stabilizers = [x ^ y for x, y in
                                    zip(self.current_stabilizers, syndrome)]
        self.syndromes = self.syndromes[self.num_ancillas:] + syndrome

    def get_new_errors(self, syndromedd):
        ''' Extracts errors from syndrome second derivative

        Input:
        syndromedd: second derivative of syndrome

        Output:
        error_list: list of (x,y) for error positions
        '''
        error_list = [j for j in range(self.num_ancillas)
                      if syndromedd[j] == 1]

        # Update list of all errors with our ones.
        self.num_errors += len(error_list)
        self.full_error_list += [(j, self.timestep-1) for j in error_list]
        return error_list

    def get_new_errors_final(self,
                             syndromedd,
                             stab_index_left,
                             stab_index_right):
        ''' Extracts errors from syndrome second derivative,
        assuming that this is a final step.

        Input:
        syndromedd: second derivative of syndrome
        stab_index_left, stab_index_right: for the final step we do
            not get the entire syndrome, so we have to take care to
            select the piece of the final step we care about.

        Output:
        error_list: list of (x,y) for error positions
        '''
        error_list = [j + stab_index_left
                      for j in range(stab_index_right - stab_index_left)
                      if syndromedd[j] == 1]

        # Update list of all errors with our ones.
        self.num_errors += len(error_list)
        self.full_error_list += [(j, self.timestep) for j in error_list]
        return error_list

    def get_weights(self, error, final_flag, **kwargs):
        '''
        Extracts weights from a pmat, and returns them.

        Input:
        error: The index of the ancilla which recorded an error.
        final_pmat: If this is None, the stored pmat will be used.
            Otherwise, we will use the pmat given (which is needed
            if this is generated for majority vote purposes).

        Output:
        weight_list: List of weights for input into blossom.
        '''

        # Initialize weight list with boundary error, and get appropriate
        # weight matrix
        if final_flag == 1:
            weight_list = [(0, self.boundary_vec_final[error])]
            if self.weight_calculation_method == 'partial_weight_matrix':
                pmat = kwargs['final_pmat']
            else:
                pmat = self.weight_matrix_final

        elif final_flag == 0:
            weight_list = [(0, self.boundary_vec[error])]
            if self.weight_calculation_method == 'partial_weight_matrix':
                pmat = self.P_mat
            else:
                pmat = self.weight_matrix

        # Sanity check
        else:
            raise ValueError('final_flag must be 0 or 1')

        # Search through full error list, add everything we find before the
        # error to the weight list
        for gi2, err2 in reversed(list(enumerate(self.full_error_list))):

            # In the case that we have multiple errors in the same round,
            # we only need to count each pair of errors once to insert
            # into the graph. This prevents double-counting.
            if err2[1] == self.timestep-1+final_flag and err2[0] >= error:
                continue
            # Calculate how far back in time we need to look
            # to see this error. As our errors are ordered, we can
            # exit if we are looking back too far without missing
            # anything.
            time_gap = self.timestep - err2[1] + final_flag - 1
            if time_gap > self.max_lookback:
                break
            # Index of second error in weight matrix
            i2 = err2[0] + time_gap * self.num_ancillas

            # Get weight out of pmat
            weight = pmat[i2, error]

            # For the partial weight matrix, this is actually still a
            # probability, so we need to convert to a weight.
            if self.weight_calculation_method == 'partial_weight_matrix':
                weight = -log(weight)
            if weight > self.max_weight:
                continue
            # Append to list.
            weight_list.append((gi2+1, weight))
        return weight_list

    def update_Pmats(self, syndrome):
        '''
        Updates the Pmats that we extract weights for our error graph
        for the partial_weight_matrix method using the approximation
                  ________________________
        P(t+1) = | P_0        (P(t)AP_0)^T|
                 | P(t)AP_0    P(t)       |
                 |________________________|

        P: the path matrix for vertices in previous timesteps
        A: the adjacency matrix for the boundary between the current
            timestep and previous,
        P_0: the path matrix for vertices in timestep t+1 *only*

        Input:
        syndrome: the measured syndrome, which allows us to extract
            what the error rate on our ancillas is

        Output:
        None
        '''

        if self.timestep == 1:  # We pre-prepare the first P_mat
            self.P_mat = sp.csc_matrix(self.Pstart_mat)
            return

        # Approximate the new P_mat from the previous one.

        # If required, pick the appropriate APO matrix based on the most-recent
        # syndrome (assuming that this is the only thing APO depends on)
        if self.symm_flag:
            AP0 = self.AP0_mats
        else:
            syn = sum([x*y for x, y in zip(syndrome, self.convert_vec)])
            AP0 = self.AP0_mats[syn]

        # AP0 extends back two timesteps, but if we have only one timestep
        # behind us then the top half of the matrix just shouldn't exist.
        if self.timestep == 2:
            # Calculate PC via the above multiplication
            PC = self.P_mat.dot(AP0[:self.num_ancillas, :])
        else:
            # Calculate PC via the above multiplication
            PC = self.P_mat.dot(AP0)

        # Filter out small entries
        PC = PC.multiply(PC >= self.prob_tol)

        # Get the size of the previous P matrix for slicing purposes
        old_size = self.P_mat.shape[0]

        # We make the new matrix in dense format, so we densify here
        PC = PC.todense()

        # start with blank slate for new matrix
        new_Pmat = np.zeros((old_size+self.num_ancillas, 2*self.num_ancillas))

        # The bottom right corner of new P matrix comes directly
        # from the previous P matrix (cutting off terms we don't)
        # need if we are far enough into the calculation
        if self.timestep == 2:
            new_Pmat[self.num_ancillas:,
                     self.num_ancillas:] = self.P_mat.todense()
        else:
            new_Pmat[self.num_ancillas:, self.num_ancillas:]\
                = self.P_mat[:, :self.num_ancillas].todense()

        # The bottom_left corner comes from the new calculation
        new_Pmat[self.num_ancillas:, :self.num_ancillas] = PC

        # The top-right corner comes from the transpose of the
        # bottom-left.
        PCt = PC[:self.num_ancillas, :].transpose()
        new_Pmat[:self.num_ancillas, self.num_ancillas:] = PCt[:, :]

        # And the top-left corner just comes from the static
        # P0 matrix
        new_Pmat[:self.num_ancillas, :self.num_ancillas] = self.P0_mat

        # Convert to sparse for better data storage.
        self.P_mat = sp.csc_matrix(new_Pmat[:self.max_rows, :])

    def get_Pmats_final(self, syndrome):
        '''
        Updates the Pmats that we extract weights for
        our error graph from, for the final round before a majority
        vote. The only difference here is the use of the final Pmats
        rather than the initial, and the fact that we return the
        Pmat rather than storing it.

        We use the approximation PC(t+1) = P(t)*A(t+1)*P0(t+1), with:

        PC: the path matrix from vertices in timestep t+1
            to timesteps previous,
        P: the path matrix for vertices in previous timesteps
        A: the adjacency matrix for the boundary between the current
            timestep and previous,
        P0: the path matrix for vertices in timestep t+1 *only*

        Input:
        syndrome: the measured syndrome, which allows us to extract
            what the error rate on our ancillas is
        error_list: list of errors in the system, so that we can
            only include rows in P_mat that will be accessed
            *** NOT CURRENTLY IMPLEMENTED***

        Output:
        new_Pmat: the PMat from which the final round of errors
            can be extracted.
        '''

        # Pick the appropriate APO matrix based on the most-recent
        # syndrome (assuming that this is the only thing APO depends on)
        if self.symm_flag:
            AP0 = self.AP0_mats_final
        else:
            syn = sum([x*y for x, y in zip(syndrome, self.convert_vec)])
            AP0 = self.AP0_mats_final[syn]

        # AP0 extends back two timesteps, but if we have only one timestep
        # behind us then the top half of the matrix just shouldn't exist.
        if self.timestep == 1:
            # Calculate PC via the above multiplication
            PC = self.P_mat.dot(AP0[:self.num_ancillas, :])
        else:
            # Calculate PC via the above multiplication
            PC = self.P_mat.dot(AP0)

        # Filter out small entries
        PC = PC.multiply(PC >= self.prob_tol)

        # Get the size of the previous P matrix for slicing purposes
        old_size = self.P_mat.shape[0]

        # We make the new matrix in dense format, so we densify here
        PC = PC.todense()

        # slice up old matrix and construct new one.
        new_Pmat = np.zeros((old_size+self.num_ancillas, 2*self.num_ancillas))

        # The top left corner of new P matrix comes directly
        # from the previous P matrix (cutting off terms we don't)
        # need if we are far enough into the calculation
        if self.timestep == 2:
            new_Pmat[self.num_ancillas:,
                     self.num_ancillas:] = self.P_mat.todense()
        else:
            new_Pmat[self.num_ancillas:, self.num_ancillas:]\
                = self.P_mat[:, :self.num_ancillas].todense()

        # The bottom-left corner comes from the new calculation
        new_Pmat[self.num_ancillas:, :self.num_ancillas] = PC

        # The top-right corner comes from the transpose of the
        # top-right.
        PCt = PC[self.num_ancillas:, :].transpose()
        new_Pmat[:self.num_ancillas, self.num_ancillas:] = PCt

        # And the top-left corner just comes from the static
        # P0 matrix
        new_Pmat[:self.num_ancillas, :self.num_ancillas] = self.P0_mat_final

        return new_Pmat
