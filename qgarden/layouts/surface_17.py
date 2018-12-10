"""surface_17 : data for the surface-17 code

Everything here is done in 'An:Dna' format - namely
ancillas are labelled as 'An' (A=X,Y,X), data qubits Dn,
and errors on data qubits Dna (a=x,y,z).
"""
from qgarden.data_structures import(
    LogicalPauli,
    LogicalClifford,
    HeisenbergFrame,
    MultiFrame,
    CodeLayout)

def get_s17_layout(boundary_label='B'):
    ancillas = {
    'X0': ['D1z', 'D2z'],
    'X1': ['D0z', 'D1z', 'D3z', 'D4z'],
    'X2': ['D4z', 'D5z', 'D7z', 'D8z'],
    'X3': ['D6z', 'D7z'],
    'Z0': ['D0x', 'D3x'],
    'Z1': ['D1x', 'D2x', 'D4x', 'D5x'],
    'Z2': ['D3x', 'D4x', 'D6x', 'D7x'],
    'Z3': ['D5x', 'D8x']
    }

    all_paulis = {p for ancilla in ancillas.values() for p in ancilla}
    ancillas[boundary_label] = [
        p for p in all_paulis if 
        len([ancilla for ancilla in ancillas.values() if 
             p in ancilla]) == 1]

    return ancillas

def swap_XZ(data):
    """Swaps x and z labels on qubits. Specific to An:Dna format.
    """
    swap_dic = {
        'X': 'Z', 'x': 'z', 'z': 'x',
        'Z': 'X', 'B': 'B', 'Y': 'Y'}
    if type(data) is dict:
        new_data = {}
        for key, val in data.items():
            new_key = swap_dic[key[0]] + key[1:]
            new_val = swap_XZ(val)
            new_data[new_key] = new_val
    elif type(data) is str:
        new_data = swap_dic[data[0]]+data[1:]
    else:
        new_data = []
        for err in data:
            if err[0] != 'D':
                raise ValueError('This only works for Dna format')
            new_err = err[:2] + swap_dic[err[2]]
            new_data.append(new_err)
    return new_data

def get_X_logical_s17():
    logical = ['D0z','D3z','D6z']
    return logical

def get_Z_logical_s17():
    logical = ['D0x','D1x','D2x']
    return logical

def get_Y_logical_s17():
    return get_X_logical_s17() + get_Z_logical_s17()

def get_XClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['X', 0]
    cliff.op_dic['Y'] = ['Y', 1]
    cliff.op_dic['Z'] = ['Z', 1]
    return cliff

def get_YClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['X', 1]
    cliff.op_dic['Y'] = ['Y', 0]
    cliff.op_dic['Z'] = ['Z', 1]
    return cliff

def get_HClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['Z', 0]
    cliff.op_dic['Y'] = ['Y', 1]
    cliff.op_dic['Z'] = ['X', 0]
    cliff.frame_list = [1, 0]
    return cliff

def get_ZClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['X', 1]
    cliff.op_dic['Y'] = ['Y', 1]
    cliff.op_dic['Z'] = ['Z', 0]
    return cliff

def get_SzClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['Y', 0]
    cliff.op_dic['Y'] = ['X', 1]
    cliff.op_dic['Z'] = ['Z', 0]
    return cliff

def get_SxClifford_sq():
    cliff = LogicalClifford()
    cliff.op_dic['X'] = ['X', 0]
    cliff.op_dic['Y'] = ['Z', 1]
    cliff.op_dic['Z'] = ['Y', 0]
    return cliff

def s17_heisenberg_frame(
        boundary_label='B',
        starting_parities=None,
        precompile=True,
        Z_first=True):
    """Makes the full Heisenberg frame for S17

    Params
    ------
    boundary_label : str
        what to call the boundary

    starting_parities : dict
        the starting parities of any logical operators
        to track.
    """

    ancillas = get_s17_layout(boundary_label)

    if Z_first:
        label_list = ['Z0','Z1','Z2','Z3',
                      'X0','X1','X2','X3',boundary_label]
    else:
        label_list = ['X0','X1','X2','X3',
                      'Z0','Z1','Z2','Z3',boundary_label]

    label_list2 = [swap_XZ(label) for label in label_list]

    pauli_strings = {
    'X': get_X_logical_s17(),
    'Y': get_Y_logical_s17(),
    'Z': get_Z_logical_s17()}

    frame1_paulis = {}
    frame2_paulis = {}
    for p, ps in pauli_strings.items():
        frame1_paulis[p] = LogicalPauli(
            ancillas, ps, precompile,
            boundary_label=boundary_label,
            dna_format=True)
        frame2_paulis[swap_XZ(p)] = LogicalPauli(
            swap_XZ(ancillas), swap_XZ(ps), precompile,
            boundary_label=boundary_label,
            dna_format=True)

    frame1 = HeisenbergFrame(
        paulis=frame1_paulis, cliffords=None, label_list=label_list)
    frame2 = HeisenbergFrame(
        paulis=frame2_paulis, cliffords=None, label_list=label_list2)

    cliffords = {
    'X': get_XClifford_sq(),
    'Y': get_YClifford_sq(),
    'Z': get_ZClifford_sq(),
    'H': get_HClifford_sq(),
    'Sz': get_SzClifford_sq(),
    'Sx': get_SxClifford_sq()}

    return MultiFrame(
        frames=[frame1,frame2], cliffords=cliffords,
        label_list=None)

def s17_code_layout():
    layout = CodeLayout(
        anc_data = [('X',[0,2]), ('X', [1,1]), ('X', [2,2]),
                    ('X', [3,1]), ('Z',[1,0]), ('Z', [1,2]),
                    ('Z', [2,1]), ('Z', [2,3])])
    return layout
