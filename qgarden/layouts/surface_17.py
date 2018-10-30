"""surface_17 : data for the surface-17 code
"""
from qgarden.data_structures import(
    LogicalPauli,
    LogicalClifford,
    HeisenbergFrame)

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
    ancillas[boundary_label] = [p for p in all_paulis if 
                     len([ancilla for ancilla in
                          ancillas.values() if p in ancilla]) == 1]

    return ancillas


def get_X_logical_s17(precompile=True, boundary_label='B'):
    logical = ['D0x','D1x','D2x']
    ancillas = get_s17_layout(boundary_label)
    return LogicalPauli(
        ancillas, logical, precompile,
        boundary_label=boundary_label,
        dna_format=True)

def get_Z_logical_s17(precompile=True, boundary_label='B'):
    logical = ['D2z','D5z','D8z']
    ancillas = get_s17_layout()
    return LogicalPauli(
        ancillas, logical, precompile,
        boundary_label=boundary_label,
        dna_format=True)

def get_Y_logical_s17(precompile=True, boundary_label='B'):
    logical = ['D0x','D1x','D2x'] + ['D2z','D5z','D8z']
    ancillas = get_s17_layout()
    return LogicalPauli(
        ancillas, logical, precompile,
        boundary_label=boundary_label,
        dna_format=True)

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
        starting_parities=None):
    """Makes the full Heisenberg frame for S17

    Params
    ------
    boundary_label : str
        what to call the boundary

    starting_parities : dict
        the starting parities of any logical operators
        to track.
    """

    paulis = {
    'X': get_X_logical_s17(boundary_label),
    'Y': get_Y_logical_s17(boundary_label),
    'Z': get_Z_logical_s17(boundary_label)}

    cliffords = {
    'X': get_XClifford_sq(),
    'Y': get_YClifford_sq(),
    'Z': get_ZClifford_sq(),
    'Sz': get_SzClifford_sq(),
    'Sx': get_SxClifford_sq()}

    parities = starting_parities or {'Z': 0}

    return HeisenbergFrame(
        paulis=paulis, cliffords=cliffords, parities=parities)

def s17_code_layout():
    layout = CodeLayout(
        anc_data = [('X',[0,2]), ('X', [1,1]), ('X', [2,2]),
                    ('X', [3,1]), ('Z',[1,0]), ('Z', [1,2]),
                    ('Z', [2,1]), ('Z', [2,3])])
    return layout
