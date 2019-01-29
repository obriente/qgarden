import numpy as np


class CodeLayout():
    def __init__(self, anc_data):
        '''Class containing the stabilizer type of each of the individual stabilizers and the position of the corresponding physical ancilla qubit in the ancilla grid layout.

        Arguments:
            anc_data {list} -- The list containing the information about each stabilizer measurement. Each entry of the list must be of the form (type, pos), where type is a string containing the information about the type of correspond stabilizer measurement (must be either 'X' or 'Z') and pos is a n-dimensional vector (array) of the position of the respective ancilla qubit on a ((d+1)**n) ancilla layout grid (with d being the distance of the code and n being the dimensionality of the code (1 for the repetition code and 2 for the surface code)). It is important to note that the individual ancilla data appearing in the array must appear in the same order as in the stabilizers givn to the adaptive weight generator/blossom decoder.

            An example of the anc_data for the distance-3 surface code (n=2), also known as Surface 17 is:
            anc_data = [('X',[0,2]), ('X', [1,1]), ('X', [2,2]), ('X', [3,1]), ('Z',[
                         1,0]), ('Z', [1,2]), ('Z', [2,1]), ('Z', [2,3])]
            where layout of the ancilla grid is the same as the one used in https://arxiv.org/pdf/1703.04136.pdf.
            A second example of a anc_data arrangement for the distance-5 repetition code (n=1) is:
            anc_data = [('Z',[0]), ('Z', [1]), ('Z', [2]), ('Z', [3])]
        '''

        self.anc_cords = []
        self.anc_types = []

        for anc_type, pos in anc_data:
            self.anc_types.append(anc_type)
            self.anc_cords.append(np.array(pos))

        self.bound_arr = self.get_bound_arr()

    def get_num_anc(self):
        '''Returns the total number of ancilla qubits in the layout

        Returns:
            int -- The total number of anc
        '''

        return len(self.anc_types)

    def get_x_stabs(self):
        '''Returns the indices of the X-type stabilizers as they appear in the full stabilizer

        Returns:
            list -- The indices of the X stabilizers
        '''

        x_stabs = [i for i, anc_type in enumerate(
            self.anc_types) if anc_type == 'X']
        return x_stabs

    def get_z_stabs(self):
        '''Returns the indices of the Z-type stabilizers as they appear in the full stabilizer

        Returns:
            list -- The indices of the Z stabilizers
        '''

        z_stabs = [i for i, anc_type in enumerate(
            self.anc_types) if anc_type == 'Z']
        return z_stabs

    def get_x_stabs_pos(self):
        '''Returns the positions of the X-type ancillas in the order as they appear in the full stabilizer

        Returns:
            list -- The positions of the X ancillas
        '''
        x_stabs = self.get_x_stabs()
        x_stabs_pos = [self.anc_cords[x_stab] for x_stab in x_stabs]
        return x_stabs_pos

    def get_z_stabs_pos(self):
        '''Returns the positions of the Z-type ancillas in the order as they appear in the full stabilizer

        Returns:
            list -- The positions of the Z ancillas
        '''
        z_stabs = self.get_z_stabs()
        z_stabs_pos = [self.anc_cords[z_stabs] for z_stabs in z_stabs]
        return z_stabs_pos

    def get_chebyshev_dist(self, start_index, end_index):
        '''Returns the manhattan distance between two ancillas positioned on a physical grid as defined by their index in the stabilizer.

        Arguments:
            from_index {int} -- The index of the first ancilla as it appears in the stabilizer
            to_index {int} -- The index of the second ancilla as it appears in the stabilizer

        Returns:
            int or None -- The Manhattan distance between the two ancillas if they belong to the same stabilizer group and None otherwise
        '''
        if self.anc_types[start_index] != self.anc_types[end_index]:
            return None
        cheb_dist = np.max(
            np.abs(self.anc_cords[start_index] - self.anc_cords[end_index]))
        return cheb_dist

    def get_bound_arr(self):
        '''Function to create a grid containing the positions of the ancillas and from there generate an array containing the information for each possible coordinate if it lies on a code boundary or not.

        Returns:
            np.array -- The boundary array. Each element of the boundary array corresponds to a position of the ancilla grid. The element is either 0 if the ancilla is not on the boundary or 1 if it is.
        '''

        dim = len(self.anc_cords[0])

        def get_grid_len(i):
            return (max(self.anc_cords, key=lambda cord: cord[i])[i] + 1)
        mat_size = [get_grid_len(i) for i in range(dim)]

        grid = np.zeros(mat_size)

        for cord in self.anc_cords:
            grid[tuple(cord)] = 1

        bound_arr = np.zeros(mat_size)

        for i in range(dim):
            shifts, overflows = [1, -1], [0, mat_size[i]-1]
            for shift, overflow in zip(shifts, overflows):
                shifted_arr = np.roll(grid, shift, axis=i)
                shifted_arr = np.delete(shifted_arr, overflow, axis=i)
                shifted_arr = np.insert(shifted_arr, overflow, 0, axis=i)
                bound_arr += shifted_arr

        bound_arr[bound_arr < (2**dim)] = 1
        bound_arr[bound_arr >= (2**dim)] = 0
        return bound_arr

    def check_boundary(self, anc_index):
        '''Checks if an ancillas as given by its index in the stabilizer measurement lies on a boundry of the code or not

        Arguments:
            anc_index {int} -- The index of the ancilla qubit from the full stabilizer measurements

        Returns:
            int -- Retruns 1 if on the boundary and 0 otherwise
        '''

        return int(self.bound_arr[tuple(self.anc_cords[anc_index])])

    def get_correction(self, start_index, end_index, stab_index_left, stab_index_right):
        '''Check whether a given error chain connecting two stabilizers constitutes a logical error or not. For the repetition and surface code, where the logical states is encoded by the total parity, constitutes a check of the number of data qubits constituting the chain (as given by the chebyshev distance)

        Arguments:
            from_index {int} -- The index of the first ancilla as it appears in the stabilizer
            to_index {int} -- The index of the second ancilla as it appears in the stabilizer
            stab_index_left {int} -- The starting index of the slice of the full stabilizers correspoding to the final measurement ones
            stab_index_right {int} -- The end index of the slice of the full stabilizers correspoding to the final measurement ones

        Returns:
            int -- 1 if the chain results in a logical error and 0 otherwise
        '''
        final_stab_range = [ind for ind in range(
            stab_index_left, stab_index_right)]
        final_stab_range.append(-1)

        if start_index not in final_stab_range or end_index not in final_stab_range:
            return 0

        if start_index == -1:
            if end_index == -1:
                return 0
            return self.check_boundary(end_index)
        if end_index == -1:
            return self.check_boundary(start_index)

        cheb_dist = self.get_chebyshev_dist(
            start_index, end_index)

        if cheb_dist is None or (cheb_dist % 2 == 0):
            return 0
        return 1
