import unittest
import numpy as np

from qgarden.layouts.surface_17 import(
    get_s17_layout,
    get_X_logical_s17,
    get_Y_logical_s17,
    get_Z_logical_s17,
    s17_heisenberg_frame)

from qgarden.data_structures import(
    LogicalPauli)


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
        ancillas = get_s17_layout()
        logicalZ = LogicalPauli(
            logical=get_Z_logical_s17(),
            ancillas=ancillas,
            precompile=False, boundary_label='B')
        logicalY = LogicalPauli(
            logical=get_Y_logical_s17(),
            ancillas=ancillas,
            precompile=False, boundary_label='B')
        logicalX = LogicalPauli(
            logical=get_X_logical_s17(),
            ancillas=ancillas,
            precompile=False, boundary_label='B')
        self.assertEqual(len(logicalZ.logical), 3)
        self.assertEqual(len(logicalX.logical), 3)
        self.assertEqual(len(logicalY.logical), 6)

    def test_surface17_boundary_connectivity(self):
        ancillas = get_s17_layout()
        logicalZ = LogicalPauli(
            logical=get_Z_logical_s17(),
            ancillas=ancillas,
            precompile=False, boundary_label='B')
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
        for label, ancilla in ancillas.items():
            print(label, ancilla)
        logicalZ = LogicalPauli(
            logical=get_Z_logical_s17(),
            ancillas=ancillas,
            precompile=True, boundary_label='B')
        logicalY = LogicalPauli(
            logical=get_Y_logical_s17(),
            ancillas=ancillas,
            precompile=True, boundary_label='B')
        logicalX = LogicalPauli(
            logical=get_X_logical_s17(),
            ancillas=ancillas,
            precompile=True, boundary_label='B')
        assert len(logicalZ.precompiled_parities) == 9
        assert len(logicalY.precompiled_parities) == 9
        assert len(logicalX.precompiled_parities) == 9

        for vertex, edges in logicalX.vertex_dic.items():
            print(vertex, edges)
        for edge, vertices in logicalX.edge_dic.items():
            print(edge, vertices)

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
        self.assertFalse(frame.paulis)
        self.assertEqual(len(frame.frames[0].paulis), 3)
        self.assertEqual(len(frame.frames[1].paulis), 3)
        self.assertEqual(len(frame.cliffords), 6)
        self.assertFalse(frame.parities)

    def test_s17_frame_parities_linked(self):
        frame = s17_heisenberg_frame()
        frame.reset(active_frame=0,parities={'Z': 0})
        self.assertEqual(frame.parities['Z'], 0)
        self.assertEqual(frame.frames[0].parities['Z'], 0)
        self.assertEqual(frame.frames[1].parities['Z'], 0)
        frame.parities['Z'] = 1
        self.assertEqual(frame.frames[0].parities['Z'], 1)
        self.assertEqual(frame.frames[1].parities['Z'], 1)

    def test_s17_frame_simple_parities(self):
        frame = s17_heisenberg_frame()
        frame.reset(active_frame=0,parities={'Z': 0})
        frame.apply_clifford('Sx')
        for key,val in frame.frames[0].paulis['Y'].precompiled_parities.items():
            print(key, val)
        print('active frame is:', frame.active_frame)
        print(frame.parities)
        frame.update(['Z0','B'])
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 1)
        frame.apply_clifford('X')
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 0)
        frame.update(['Z0','Z1'])
        self.assertEqual(len(frame.parities), 1)
        self.assertEqual(frame.parities['Y'], 0)

    def test_s17_superframe(self):
        frame = s17_heisenberg_frame()
        frame.reset(active_frame=0,parities={'Z': 0})
        frame.apply_clifford('H')
        self.assertEqual(frame.active_frame, 1)
        self.assertEqual(frame.parities['X'], 0)

    def test_s17_moreops(self):
        frame = s17_heisenberg_frame()
        frame.reset(active_frame=0,parities={'Z': 0})
        frame.apply_clifford('H')
        print(frame.parities)
        print(frame.active_frame)
        print(frame.frames[1].parities)
        frame.update(['X1', 'B'])
        print(frame.frames[1].parities)
        frame.apply_clifford('H')
        self.assertEqual(frame.parities['Z'],1)

    def test_s17_labelled(self):
        frame = s17_heisenberg_frame()
        frame.reset(active_frame=0,parities={'Z': 0})
        frame.update_from_index(1,-1)
        frame.apply_clifford('H')
        print(frame.parities)
        print(frame.active_frame)
        print(frame.frames[1].parities)
        frame.update_from_index(1,-1)
        print(frame.frames[1].parities)
        frame.apply_clifford('H')
        self.assertEqual(frame.parities['Z'],0)
