"""Code to store data structures for graphs"""


class Graph:
    """Graph underlying graph class. Defines any methods
    to be made generic. 
    """
    def __init__(self):
        pass

    def get_distance(self,v1,v2):
        """Return the distance between any two vertices
        Params
        ------
        v1, v2 : hashable types
            labels for the two vertices

        Returns
        ------
        distance : the distance between the two vertices
        """

        raise NotImplementedError


class PointerGraph(Graph):
    """PointerGraph - a structure for a graph that stores
    a list of vertices for each edge, and a list of edges
    for each vertex.

    Params
    ------
    vertex_dic : dict
        dictionary containing a list of all the edges
        connected to each named vertex
    edge_dic : dict
        dictionary containing a list of all the vertices
        connected to each named edge
    """

    def __init__(
            self,
            vertex_dic=None,
            edge_dic=None,
            **kwargs):

        self.vertex_dic = vertex_dic or {}
        self.edge_dic = edge_dic or {}
        super().__init__(**kwargs)

    def make_vertex_dic(self):
        """Constructs the dictionary of edges pointing to each
        vertex from the dictionary of vertices pointing to each
        edge.
        """
        vertex_set = set([
            vertex for vertices in self.edge_dic.values()
            for vertex in vertices])

        # tongue-twisting code.
        self.vertex_dic = {
            vertex: [edge for edge, vertices in
                     self.edge_dic.items()
                     if vertex in vertices]
            for vertex in vertex_set
        }

    def make_edge_dic(self):
        edge_set = set([
            edge for edges in self.vertex_dic.values()
            for edge in edges])

        self.edge_dic = {
            edge: [vertex for vertex, edges in
                   self.vertex_dic.items()
                   if edge in edges]
            for edge in edge_set
        }


    def get_distance(self,v1,v2):
        """Return the distance between any two vertices
        Params
        ------
        v1, v2 : hashable types
            labels for the two vertices

        Returns
        ------
        distance : the distance between the two vertices
        """
        connected_to_v1 = {v1}
        distance = 0
        while True:
            distance += 1
            new_edges = set([edge for vertex in connected_to_v1 
                             for edge in self.vertex_dic[vertex]])
            new_vertices = set([vertex for edge in new_edges
                             for vertex in self.edge_dic[edge]])
            if v2 in new_vertices:
                return distance

            if new_vertices.issubset(connected_to_v1):
                raise RuntimeError('Vertices are not connected')


class WeightedGraph(Graph):
    """WeightedGraph - an extension of a graph that allows
    edges and/or vertices to be weighted.

    Params
    ------
    edge_weights: dict
        list of weights on edges
    vertex_weights: dict
        list of weights on vertices
    """
    def __init__(
            self, 
            edge_weights=None, 
            vertex_weights=None,
            **kwargs):

        self.edge_weights = edge_weights or {}
        self.vertex_weights = vertex_weights or {}
        super().__init__(**kwargs)

    def combine_weights(self,w1,w2,method):
        """Gives various methods for combining weights

        Params
        ------
        w1, w2 : float
            the weights to be combined together (i.e. weights
            on two paths that meet at a single vertex)
        method : 'sum', 'min', 'expect_same', 'prod', 'xor'
            the method to combine weights

        Raises
        ------
        ValueError: if method is not understood 
        """
        if method == 'min':
            return min(w1,w2)
        elif method == 'expect_same':
            return w1
        elif method == 'check_same':
            if w1 != w2:
                raise ValueError('Found different weights')
            return w1
        elif method == 'sum':
            return w1+w2
        elif method == 'prod':
            return w1*w2
        elif method == 'xor':
            return w1^w2
        else:
            raise ValueError(
                "I can only combine weights via "+
                "'sum', 'min', 'except_same', 'xor', 'prod'")


class WeightedPointerGraph(WeightedGraph, PointerGraph):
    """Combination of the WeightedGraph and PointerGraph
    classes    
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_weight_min_distance(
            self,v1,v2,
            string_by='sum',
            combine_by='min',
            assert_connected=False):
        """Gets the path weight of the shortest path between
        two vertices - i.e. the combination of the edge
        weights over the path.

        Params
        ------
        v1, v1: hashable
            labels of the two vertices
        string_by: 'sum', 'prod' or 'xor'
            determines how weights on subsequent edges are
            combined - either by adding or multiplying or xor.
        combine_by: 'min', 'expect_same', 'check_same', 'sum'
            determines what the function does if multiple paths
            are the shortest - either take the minimum,
            expect the same result (and raise an error 
            otherwise), or add the weights.

        Returns
        ------
        weight : float
            weight of the shortest path between two vertices.

        Raises
        ------
        ValueError
            if action_if_compete ='expect_same' and two paths
            do not have the same weight.
        RuntimeError
            if assert_connected is true and vertices are not
            actually connected.
        ValueError
            if string_by is not 'sum', 'xor' or 'prod'
        """
        if string_by == 'sum' or string_by == 'xor':
            path_weights = {v1:0}
        elif string_by == 'prod':
            path_weights = {v1:1}
        else:
            raise ValueError(
                "I can only string weights via "+
                "'sum', 'prod', 'xor")

        while True:
            new_path_weights = self.step_weights(
                path_weights, string_by, combine_by)

            if v2 in new_path_weights:
                return new_path_weights[v2]

            if set(new_path_weights.keys()).issubset(
                    set(path_weights.keys())):
                if assert_connected:
                    raise RuntimeError('Vertices are not connected')
                return None

            path_weights = {**path_weights, **new_path_weights}


    def step_weights(self, path_weights, string_by, combine_by):
        """Single step of the path-finding algorithm.

        Params
        ------
        path_weights : dict
            The dictionary of the combined weight of the paths
            from the initial vertex to all other vertices.
        string_by: 'sum', 'prod' or 'xor'
            determines how weights on subsequent edges are
            combined - either by adding or multiplying or xor.
        combine_by: 'min', 'expect_same', 'sum'
            determines what the function does if multiple paths
            are the shortest - either take the minimum,
            expect the same result (and raise an error 
            otherwise), or add the weights.
        """
        new_path_weights = {}
        for vertex,vw in path_weights.items():
            for edge in self.vertex_dic[vertex]:
                ew = self.edge_weights[edge]
                new_weight = self.combine_weights(
                    vw,ew,string_by)
                for end_vertex in self.edge_dic[edge]:

                    if end_vertex == vertex:
                        continue

                    elif end_vertex in new_path_weights:
                        new_weight = self.combine_weights(
                            new_weight, 
                            new_path_weights[end_vertex],
                            combine_by)

                    elif end_vertex in path_weights:
                        new_weight = self.combine_weights(
                            new_weight,
                            path_weights[end_vertex],
                            combine_by)

                    new_path_weights[end_vertex] = new_weight
        return new_path_weights
