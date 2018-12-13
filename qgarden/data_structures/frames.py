"""Classes to store frames - guesses from a decoder about the
logical state of the system.
"""
from .graph import(
    PointerGraph,
    WeightedPointerGraph)


class HeisenbergFrame:
    """A Heisenberg frame tracks the effect of errors on a final
    logical parity measurement. As a decoder reports errors
    occurring, the frame checks whether those errors commute
    or anti-commute with its chosen logical operators at this
    given time, and updates itself accordingly. Then, when
    a final measurement occurs, the frame decides whether
    the bit or the not of the bit should be taken.

    Because we assume fault-tolerance (even when we dont have it,
    we can't do any better), an error chain that ends at time t
    should be equivalent to an error chain that occurs entirely
    at time t (except for measurement errors reaching back in time).
    As such, the frame does not itself need to keep track of time.

    Params
    ------
    parities : dict of booleans
        a list of parities for every logical pauli
        being tracked.
    paulis : dict of LogicalPaulis:
        a list of logical operators
    operators : dict of LogicalPaulis
        a list of logical operators
    cliffords : dict of LogicalCliffords
        a list of logical clifford operators (i.e. the update
        rules).
    """

    def __init__(
            self, paulis, cliffords, parities=None,
            label_list=None):
        self.parities = parities or {}
        self.paulis = paulis
        self.cliffords = cliffords
        self.label_list = label_list

    def update(self, error):
        """Updates frame with detected error

        Params
        ------
        error : list in the form [s1,s2]
            the stabilizers between which the error occurred
        """
        for op_label in self.parities:
            self.parities[op_label] ^=\
                self.paulis[op_label].anticommutes_with(error)

    def update_from_index(self, e1, e2):
        """Updates from a pair of indices instead of labels
        
        Params
        ------
        e1, e2 : int
            the indices corresponding to the errors of interest.
        """
        if self.label_list is None:
            raise ValueError('I need a label_list to use this.')

        error = [self.label_list[e1], self.label_list[e2]]
        self.update(error)

    def apply_clifford(self, label):
        """Applies logical Clifford operator to the frame

        Params
        ------
        logical : string
            an update rule for the logical gates.
        """
        clifford = self.cliffords[label]
        new_parities = {}
        for op_label, parity in self.parities.items():
            new_op_label, d_parity = clifford.update(op_label)
            new_parities[new_op_label] = parity ^ d_parity
        self.parities = new_parities

    def get_parity(self, measurement):
        """Gets the parity of a measurement.
        Params
        ------
        measurement : hashable
            name of the measurement to take

        Returns
        -------
        parity : None or boolean
            the result of the measurement. If untracked,
            we return None (to be taken as that the starting
            state was not an eigenstate).
        """
        if measurement not in self.parities:
            return None
        return self.parities[measurement]

    def reset(self, parities):
        self.parities = parities


class MultiFrame(HeisenbergFrame):
    """A multiframe is a class that contains multiple Heisenberg
    frames that act on the same set of logical operators.

    In particular, a Heisenberg frame contains a set representation
    of the stabilizers; this allows for that representation to change.

    Params
    ------
    frames : list of HeisenbergFrames
        the set of HeisenbergFrames.
    active_frame : which frame is currently active
    """
    def __init__(self, frames, 
                 active_frame=0,
                 parities=None, **kwargs):
        self.frames = frames
        self.active_frame = active_frame
        super().__init__(paulis=None, **kwargs)
        if parities:
            self.reset(active_frame, parities)

    def reset(self, active_frame, parities):
        """Reset the parity in all frames
        
        Params
        ------

        parity : dict
            the reset parities of all logical operators
        """
        self.active_frame = active_frame
        self.parities = parities
        for frame in self.frames:
            frame.reset(parities)

    def apply_clifford(self, label):
        """Apply a clifford operator, which involves updating
        the frame and then applying the clifford operator to the
        logical operators.

        Params
        ------
        error : list in the form [s1,s2]
            the stabilizers between which the error occurred
        label : str
            the label of the Clifford to be updated.
        """
        self.active_frame =\
            self.cliffords[label].update_frame(self.active_frame)
        super().apply_clifford(label)
        self.frames[self.active_frame].reset(self.parities)

    def update(self, error):
        """Passes error through to the appropriate
        frame for update

        Params
        ------
        error : list in the form [s1,s2]
            the stabilizers between which the error occurred
        """
        self.frames[self.active_frame].update(error)

    def update_from_index(self, e1, e2):
        """Passes error through to the appropriate
        frame for update

        Params
        ------
        e1, e2 : int
            the indices corresponding to the errors of interest.
        """
        self.frames[self.active_frame].update_from_index(e1,e2)

# format_dependent commutation check
def check_sign_single_pauli(sp1, sp2):
    '''
    Checks the sign between two single qubit
    Pauli operators written in the form ['Dna'],
    where D = data qubit, n = qubit index, 
    and a=x,y,z denotes the Pauli.

    sign(P1,P2) = 0 if [P1,P2] = 0
    otherwise sign(P1,P2) = 1 and {P1,P2} = 0.

    Params
    ------
    sp1, sp2 : str
        strings of the Pauli operators to check.
    '''
    if sp1[0] != 'D' or sp1[0] != 'D':
        raise ValueError('Paulis must be in form Dna!')
    if sp1[1] == sp2[1] and sp1[2] != sp2[2]:
        return 1
    return 0

def check_sign_multi_pauli(p1,p2):
    '''
    Checks the sign between two Pauli operators written
    in the form ['Dn_0a_0','Dn_1a_1',...] where
    n_i = qubit index, a_i = x,y,z of single Pauli operator.

    sign(P1,P2) = 0 if [P1,P2] = 0
    otherwise sign(P1,P2) = 1 and {P1,P2} = 0.

    Params
    ------
    p1, p2 : list of strings
        Pauli operators written in tensor factor form described
        above.
    '''

    # Code is a bit of a hack as it checks each pair of
    # tensor factors rather than ordering and comparing them.
    sign = 0
    for sp1 in p1:
        for sp2 in p2:
            sign = sign ^ check_sign_single_pauli(sp1,sp2)
    return sign


class AncillaGraph(PointerGraph):
    """A graph where edges correspond to data qubits and
    vertices to ancilla qubits
    """
    def __init__(self, ancillas, boundary_label,
            dna_format=False):
        """
            Params
            ------
            ancillas : dict
                for each ancilla qubit, a list of the type
                of errors measured on each data qubit.
                e.g. {'X0':['D0z','D1z'], 'Z1':['D0x','D1x']}
            boundary_label : str
                the label of the boundary vertex in the ancilla
                graph
            dna_format : boolean
                whether the labels are in the format 'Dna'.
        """
        super().__init__(vertex_dic=dict(ancillas))
        self.dna_format = dna_format
        self.boundary_label = boundary_label
        self.make_edge_dic()
        # Edges to the boundary are one way only,
        # the return pointers.
        self.vertex_dic[boundary_label] = []
        if dna_format:
            for ancilla in ancillas.values():
                self.check_commutes_with_ancillas(ancilla)

    def check_commutes_with_ancillas(self, op):
        '''
        Checks if an operator commutes with all
        of the ancillas stored in the graph.

        Params
        ------
        op: string of form 'Dna'
            the operator to be checked.
        '''
        for ancilla in self.vertex_dic.values():
            if check_sign_multi_pauli(op,ancilla) != 0:
                raise ValueError('{} does not commute with {}'.format(
                                op, ancilla))


class LogicalPauli(AncillaGraph, WeightedPointerGraph):
    """A weighted graph generated by a set of ancilla
    qubits and a logical operator defined on a set of
    Paulis.

    Params
    ------
    ancillas : dict of lists of str
        the ancillas that measure this operator
    logical : list of str
        the logical operator this Pauli describes
    precompile : boolean
        whether to precompile a table of parities
        or calculate on the fly.
    """
    def __init__(self, ancillas, logical,
                 precompile=False, **kwargs):

        super().__init__(ancillas, **kwargs)
        self.logical = logical
        self.precompile = precompile

        self.edge_weights = {}
        for pauli in self.edge_dic:
            if pauli in logical:
                self.edge_weights[pauli] = 1
            else:
                self.edge_weights[pauli] = 0

        if precompile:
            self.precompiled_parities = {s1:
                {s2: self.get_weight_min_distance(
                    s1, s2, 'xor', 'expect_same') 
                 for s2 in self.vertex_dic if s2 != s1}
                for s1 in self.vertex_dic}
            # Fix boundary as it doesn't see
            # any paths out.
            self.precompiled_parities[self.boundary_label] = {
                key: self.precompiled_parities[key][self.boundary_label]
                for key in self.precompiled_parities if
                key != self.boundary_label
            }

    def anticommutes_with(self, error):
        s1,s2 = error
        if self.precompile:
            return self.precompiled_parities[s1][s2]
        else:
            return self.get_weight_min_distance(
                s1,s2,'xor','expect_same')


class LogicalClifford:
    """LogicalClifford: update rules for Pauli operators
    """

    def __init__(self, op_dic=None, frame_list=None):
        self.op_dic = op_dic or {}
        self.frame_list = frame_list or []

    def update(self, op_label):
        return self.op_dic[op_label]
    def update_frame(self, frame_index):
        if self.frame_list:
            return self.frame_list[frame_index]
        return frame_index
