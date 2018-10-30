import unittest
from qgarden.data_structures import(
    Graph,
    PointerGraph,
    WeightedGraph,
    WeightedPointerGraph)

from qgarden.data_structures import(
    AncillaGraph,
    check_sign_single_pauli,
    check_sign_multi_pauli,
    HeisenbergFrame,
    LogicalPauli,
    LogicalClifford)


class TestGraphs(unittest.TestCase):

    def test_Graph_init(self):
        g = Graph()
        with self.assertRaises(NotImplementedError):
            g.get_distance(None,None)

    def test_PointerGraph_init(self):
        g = PointerGraph()
        self.assertTrue(isinstance(g.vertex_dic,dict))
        self.assertTrue(isinstance(g.edge_dic,dict))

    def test_distance(self):
        vertex_dic = {
        'v0': ['e0'],
        'v1': ['e0']}
        edge_dic = {
        'e0': ['v0','v1']}
        g = PointerGraph(vertex_dic,edge_dic)
        self.assertTrue('v0' in g.vertex_dic)
        self.assertTrue('v1' in g.vertex_dic)
        self.assertTrue('e0' in g.edge_dic)
        self.assertEqual(g.get_distance('v0','v1'), 1)
        self.assertEqual(g.get_distance('v1','v0'), 1)

    def test_graph_population(self):
        vertex_dic = {
        'v0': ['e0'],
        'v1': ['e0']}
        edge_dic = {
        'e0': ['v0','v1']}
        g1 = PointerGraph(vertex_dic=vertex_dic)
        g1.make_edge_dic()
        self.assertTrue('e0' in g1.edge_dic)
        self.assertEqual(len(g1.edge_dic['e0']), 2)
        self.assertTrue('v0' in g1.edge_dic['e0'])
        self.assertTrue('v1' in g1.edge_dic['e0'])
        g2 = PointerGraph(edge_dic=edge_dic)
        g2.make_vertex_dic()
        self.assertTrue('v0' in g2.vertex_dic)
        self.assertTrue('v1' in g2.vertex_dic)
        self.assertEqual(len(g2.vertex_dic['v0']), 1)
        self.assertEqual(len(g2.vertex_dic['v1']), 1)
        self.assertTrue('e0' in g2.vertex_dic['v0'])
        self.assertTrue('e0' in g2.vertex_dic['v1'])


    def test_WeightedGraph_init(self):
        g = WeightedGraph()
        self.assertTrue(isinstance(g.edge_weights,dict))
        self.assertTrue(isinstance(g.vertex_weights,dict))

    def test_combine_weights(self):
        g = WeightedGraph()
        self.assertEqual(g.combine_weights(1,1,'sum'),2)
        self.assertEqual(g.combine_weights(1,1,'prod'),1)
        self.assertEqual(g.combine_weights(1,1,'xor'),0)
        self.assertEqual(g.combine_weights(1,2,'min'),1)
        self.assertEqual(g.combine_weights(1,1,'expect_same'),1)
        with self.assertRaises(ValueError):
            g.combine_weights(1,2,'check_same')

    def test_WeightedPointerGraph_init(self):
        g = WeightedPointerGraph()
        self.assertTrue(isinstance(g, WeightedGraph))
        self.assertTrue(isinstance(g, PointerGraph))
        self.assertTrue(isinstance(g.edge_weights, dict))
        self.assertTrue(isinstance(g.vertex_weights, dict))
        self.assertTrue(isinstance(g.vertex_dic, dict))
        self.assertTrue(isinstance(g.edge_dic, dict))

    def test_weight_min_distance(self):
        g = WeightedPointerGraph()
        g.vertex_dic['v0'] = ['e0']
        g.vertex_dic['v1'] = ['e0', 'e1']
        g.edge_dic['e0'] = ['v0','v1']
        g.vertex_dic['v2'] = ['e1']
        g.edge_dic['e1'] = ['v1','v2']
        g.edge_weights['e0'] = 1
        g.edge_weights['e1'] = 1
        self.assertEqual(g.get_weight_min_distance('v0','v2'), 2)
        self.assertEqual(
            g.get_weight_min_distance(
                'v2','v0','xor','expect_same'), 0)

    def test_nontrivial_xor(self):
        g = WeightedPointerGraph()
        g.vertex_dic['v0'] = ['e0','e1']
        g.vertex_dic['v1'] = ['e0','e3']
        g.vertex_dic['v2'] = ['e1','e2']
        g.vertex_dic['v3'] = ['e2','e3']
        g.make_edge_dic()
        g.edge_weights['e0'] = g.edge_weights['e2'] = 1
        g.edge_weights['e1'] = g.edge_weights['e3'] = 0
        self.assertEqual(
            g.get_weight_min_distance(
                'v0', 'v3', 'xor', 'expect_same'), 1)

class TestAncillaGraph(unittest.TestCase):

    def test_init(self):
        graph = AncillaGraph(ancillas = {'test': ['D0z', 'D1z'], 'test2': ['D1z','D2z']},
            boundary_label='blah', dna_format=True)
        self.assertTrue(graph.dna_format)
        self.assertEqual(len(graph.vertex_dic['blah']), 0)
        self.assertEqual(len(graph.edge_dic), 3)
        self.assertEqual(len(graph.vertex_dic), 3)
        self.assertEqual(graph.boundary_label, 'blah')

class TestLogicalPauli(unittest.TestCase):

    def test_init(self):
        lp = LogicalPauli(ancillas = {},
            logical=[], boundary_label='B')
        self.assertFalse(lp.logical)
        self.assertEqual(len(lp.vertex_dic), 1)
        self.assertEqual(len(lp.edge_dic), 0)
        self.assertFalse(lp.precompile)


class TestHeisenbergFrame(unittest.TestCase):

    def test_init(self):
        paulis = 'test'
        parities = 'test2'
        cliffords = 'test3'
        frame = HeisenbergFrame(paulis,parities,cliffords)
        self.assertEqual(paulis, 'test')
        self.assertEqual(parities, 'test2')
        self.assertEqual(cliffords, 'test3')

    def test_update(self):
        class MockPauli:
            def anticommutes_with(self,error):
                return True
        frame = HeisenbergFrame(paulis={'test':MockPauli()},
                                cliffords=None,
                                parities={'test':0})
        frame.update(error=None)
        self.assertEqual(frame.parities['test'], 1)

    def test_clifford(self):
        class MockClifford:
            def update(self,label):
                if label == 'test':
                    return 'test2', 1
                else:
                    raise ValueError
        frame = HeisenbergFrame(paulis=None,
                                cliffords={'cliff': MockClifford()},
                                parities={'test': 0})
        frame.apply_clifford('cliff')
        self.assertEqual(frame.parities['test2'], 1)

    def test_get_parities(self):
        frame = HeisenbergFrame(paulis=None,cliffords=None,parities={'test': 0})
        self.assertEqual(frame.get_parity('test'), 0)
        self.assertFalse(frame.get_parity('other'))
