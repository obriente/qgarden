'''Example_simple_decoder

Creates a simple decoder and runs it on a toy example syndrome
'''

from qgarden.decoder_template_simple import run
from qgarden.layouts.surface_17 import s17_code_layout


# Presetting environment variables
distance = 3  # Code distance

# Error rates - for debugging setting all of these
# to 0.01 should be reasonable. For real-world testing
# using the weight_gen_adaptive is preferable.
px = 0.01
py = 0.01
pz = 0.01
pm = 0.01

# Code layout - unique to surface-17
code_layout = s17_code_layout()

# max_lookback - how far back to look in time to connect
# errors - set to 1 or 2
max_lookback = 2

# Data generation
# index 1 - different experiments
# index 2 - ancilla measurements vs final stabilizers
# index 3 (ancillas) - timestep
# index 4 (ancillas) - ancilla index
# index 3 (final stabilizers) - stabilizer index
syn = [[0]*8, [0]*8, [0]*8, [0]*8]
fstab = [0]*4
data = [[syn, fstab], ]

# Lets make it slightly non-trivial
# This is the syndrome corresponding to a data qubit error
# on one of the data qubits that only flips a single Z
# stabilizer.
data[0][0][2][4] = 1
data[0][0][3][4] = 0
data[0][1][0] = 1


# Output - a single parity bit
print(run(data, distance, max_lookback, px, py, pz, pm, code_layout))
