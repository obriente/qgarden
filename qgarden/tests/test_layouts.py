import unittest
import numpy as np

from qgarden.layouts.surface_17 import(
    get_s17_layout,
    get_X_logical_s17,
    get_Y_logical_s17,
    get_Z_logical_s17,
    s17_heisenberg_frame)


class TestSurface17(unittest.TestCase):

    def test_surface17_ancillas(self):
        ancillas = get_s17_layout()
        self.assertEqual(len(ancillas), 9)
        self.assertEqual(len(ancillas['B']), 12)
        all_paulis = {p for ancilla in ancillas.values() for p in ancilla}
        assert(len(all_paulis) == 18)
        for p in all_paulis:
            self.assertEqual(len([ancilla for ancilla in ancillas.values()
                                  if p in ancilla]), 2)

    def test_surface17_operators(self):
        logicalZ = get_Z_logical_s17(precompile=False)
        logicalY = get_Y_logical_s17(precompile=False)
        logicalX = get_X_logical_s17(precompile=False)
        self.assertEqual(len(logicalZ.logical), 3)
        self.assertEqual(len(logicalX.logical), 3)
        self.assertEqual(len(logicalY.logical), 6)

    def test_surface17_boundary_connectivity(self):
        logicalZ = get_Z_logical_s17(precompile=False)
        for err, ancillas in logicalZ.edge_dic.items():
            print(err, ancillas)
        for ancilla, errs in logicalZ.vertex_dic.items():
            print(ancilla, errs)
        ancillas = get_s17_layout()
        for ancilla in ancillas:
            if ancilla == 'B':
                continue
            print('trying ancilla: ', ancilla)
            val = logicalZ.get_weight_min_distance(
                ancilla, 'B', string_by='xor', combine_by='expect_same', 
                assert_connected=True)
            self.assertTrue(val==0 or val==1)

    def test_surface17_precompilation(self):
        ancillas = get_s17_layout()
        logicalZ = get_Z_logical_s17()
        logicalY = get_Y_logical_s17()
        logicalX = get_X_logical_s17()
        assert len(logicalZ.precompiled_parities) == 9
        assert len(logicalY.precompiled_parities) == 9
        assert len(logicalX.precompiled_parities) == 9

        for a1 in ancillas:
            for a2 in ancillas:
                if a1 == a2:
                    continue
                for name, logical in [['X',logicalX],['Y',logicalY],['Z',logicalZ]]:
                    print('Trying: ', a1, a2, name)
                    print(logical.precompiled_parities[a1][a2])
                    self.assertTrue(
                        ((a1[0] == a2[0] or a1[0] == 'B' or a2[0] == 'B') and
                            (logical.precompiled_parities[a1][a2] == 0 or
                                logical.precompiled_parities[a1][a2] == 1)) or
                        ((a1[0] != a2[0] and a1[0] != 'B' and a2[0] != 'B') and
                            ((logical.precompiled_parities[a1][a2]) is None)))

    def test_s17_frame(self):
        frame = s17_heisenberg_frame()
        self.assertEqual(len(frame.paulis), 3)
        self.assertEqual(len(frame.cliffords), 6)
        self.assertEqual(len(frame.parities), 1)

    def test_s17_frame_simple_parities(self):
        frame = s17_heisenberg_frame()
        frame.apply_clifford('Sx')
        frame.update(['Z0','B'])
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 1)
        frame.apply_clifford('H')
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 0)
        frame.update(['Z0','Z1'])
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 0)
