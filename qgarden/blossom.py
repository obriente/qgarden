"""
Blossom_v3_2

Written by Thomas O'Brien, after discussions with Xiang Fu and Adriaan Rol

3.2: improved variable names

Implements the blossom algorithm in a manner similar to that outlined
In the blossom V paper by Vladimir Kolmogorov, but using only
single alternating trees.

Supports adding single vertices at a time, and memory management.
A wrapper has been added to insert an entire weight matrix at once,
which is in the tests folder of the quantumerrorcorrection package.
"""
#Python 2.7 import
#from __future__ import division
import numpy as np
from collections import deque
# Python 2.7 import
#import Queue as queue
# Python 3 import
import queue


class Bloss:

    # INITIALISATION ###
    def __init__(self, tbw):
        """
        Initializing graph with only boundary term.

        Input:

        tbw: weight cutoff for the time boundary
            every vertex weight is not allowed to grow
            beyond n*tbw, where n is the number of time
            steps to the boundary.

        Output: none
        """

        # Initialise all permanent variables here; both for realistic and
        # investigation purposes.

        #
        # Structures to deal with time edge
        #

        # time_boundary_weight: a hypothetical weight
        # connecting vertices to the time boundary (i.e. a vertex
        # n timesteps before the time boundary would have an edge
        # of weight n*tbw that it can't decide whether or not to
        # connect with). If we ever hit such an edge, the program
        # halts until either the time-step increases or we are
        # forced to finish, in which case we make explicit connections
        # to the time boundary
        self.time_boundary_weight = tbw

        # time_weight_list: list of 'weights' between a vertex
        # and the time-boundary, giving us a flag to stop calculating
        self.time_weight_list = [None]

        # time_boundary: stores the position of the time boundary
        # when it is set (otherwise empty)
        self.time_boundary = None

        #
        # Structures for conversion / return to user.
        #

        # pairing_real_list: prl[ril[i]] = pl[i]
        # I.e. this is what gets returned to the user. Using two lists
        # is solely to prevent unnecessary additional memory access
        # On the FPGA, this is not guaranteed to be up to date
        # unless updated explicitly.
        self.pairing_real_list = [None]

        # reverse_index_list: Allows the updating of self.pairing_real_list
        # from self.pairing_list
        self.reverse_index_list = [0]

        # forward_index_list: list storing the order in which vertices
        # were inserted; when we need to free up memory this is accessed from
        # the top to give the order in which vertices should be deleted
        self.forward_index_list = [0]

        #
        # Memory management structures
        #

        # weight_cap: stores the weight of the lowest weight edge from
        # tree to an unpaired vertex - this gives a maximum size on our
        # instruction list
        self.weight_cap = None

        # _err: error margin for weight cap checking: In the case of
        # degeneracy, python does stupid things with checking >. So, add a
        # small error margin to the weight cap every time we check it.
        self._err = 0.00001

        #
        # Structures storing information about the graph
        #

        # pairing_list: pl[i] is a pair [v,e], where
        # v is the vertex that vertex i is paired to, and
        # e is the edge that they are paired along.
        self.pairing_list = [[]]

        # unpaired_list: list of vertices to become roots,
        # to be accessed in a lifo manner (so in python we use a deque)
        self.unpaired_list = deque([])  # O(N)

        # node_weight_list: so we can find
        # each effective edge weight
        self.node_weight_list = [0]

        # node_edge_list: stores a list of edges each node is connected to
        # each entry takes the form [pointer, index], with:
        #
        # pointer: a pointer to self.edge_list for the edge
        #
        # index: a bit telling whether the other side's indices
        # are stored in self.edge_list[pointer][1] and self.edge_list[pointer][3]
        # or self.edge_list[pointer][2] and self.edge_list[pointer][4]
        # this reduces access time slightly.
        self.node_edge_list = [[]]  # O(N)

        # edge_list: stores the list of edges
        # Each edge takes the form [weight,v1,v2,n1,n2,active_flag] with:
        #
        # Weight: the delayed effective weight
        #   (edge weight - all unactive node weights)
        #
        # (v1,v2): the vertex pair, with v1 < v2
        #
        # (n1,n2): is the pair of highest nodes that this edge is active for.
        #           (correlated with v1, v2)
        #
        # active_flag: a flag for whether both nodes this edge connects are
        #           active (not contained within a blossom)
        #
        # We do not keep track of the nodes of unactive edges.
        self.edge_list = []

        #
        # Structures storing information about blossoms
        #

        # blossom_list: list of blossoms if present
        # i.e. the blossoms which a vertex/blossom sit inside.
        # Note that a node should only ever be contained within a tree
        # if self.blossom_list[j]=None
        self.blossom_list = [None]

        # root_list: list of roots of cycles if present.
        self.root_list = [None]

        # cycle_list: list of cycles if present - i.e. this
        # is only not None if we have a blossom, in which case it is
        # the list of the indices it contains.
        self.cycle_list = [[]]  # O(B)

        # cycle_edge_list: list of cycle edges if blossom
        # in form [ep, vp]. cel[i] is the edge connecting cl[i] and cl[i+1]
        # and vp is the pointer to go to cl[i+1]
        self.cycle_edge_list = [[]]

        #
        # Structures storing information about trees
        #

        # instruction_list: list of instructions for the algorithm.
        # self-updating (but currently empty as we have no tree).
        # Instructions come in the form
        # [weight,type,{edge index/blossom index},[vertex pointer]]
        #
        #   weight = edge weight - sum(nodes) node weight + tree weight
        #           or for a blossom blossom_weight + tree_weight
        #
        #   type = 0 for collapsing tree
        #          1 for adding to tree
        #          2 for making blossom
        #          3 for breaking a blossom
        #      (this is the best priority order for degenerate instructions;
        #       collapsing trees are the best thing to do, and adding vertices
        #       to trees is a much simpler operation than making blossoms.
        #       We make blossoms before breaking them because this can mean the
        #       breaking is unnecessary. Note that this does increase memory
        #       use (as we can have blossoms of zero weight sitting inside
        #       other blossoms).
        #
        #   edge index/blossom index: the index to the thing we want to do
        #       stuff to
        #
        #   vertex pointer: the pointer to the vertex not in the tree in the
        #       for types 0/1/2.

        self.instruction_list = None  # O(T)

        # total_growth: the amount a tree has grown; allows for
        # comparison between two instructions without continuous update.
        self.total_growth = 0

        # tree_branch_list: list correlated with self.node_weight_list that stores a
        # triplet of data; the node that this node is paired to, the index of
        # the paired edge, and the vp of the edge pointing in the direction
        # away from the root (i.e. edge[4-vp] goes towards root)
        # edge within self.edge_list[self.tree_branch_list[0]].

        # Given that tbl[j][0] = self.edge_list[tbl[j][1]][3+tbl[j][2]], we might
        # consider removing it; we trade immediacy of access for not having
        # to keep it updated.
        self.tree_branch_list = [[]]  # O(N)

        # outer_node_list: list of all outer nodes in tree
        self.outer_node_list = []  # O(T)

        # inner_node_list: list of all inner nodes
        self.inner_node_list = []  # O(T)

    ###########################################################################
    # Top level algorithms (algorithms to be called from outside this file) ###
    ###########################################################################

    def add_t_weight(self):
        '''
        Adds one timestep of weight to the currently-present vertices

        Input: None

        Output: None
        '''

        # Go over all vertices, and increment their time weight
        for j in range(len(self.time_weight_list)):
            # If the vertex exists and is not the physical boundary
            if self.time_weight_list[j]:
                self.time_weight_list[j] += self.time_boundary_weight

    def run(self):
        '''
        Pulls instructions from the instruction list, and sends them to
        the appropriate function to be executed. Upon hitting the time
        boundary or running out of vertices to pair, exits. Creates
        trees whenever trees are collapsed and new ones not formed.

        Input: None

        Output: None

        '''

        # If there are no unpaired vertices, return
        if not self.unpaired_list and not self.instruction_list:
            return

        # Make a new tree if we don't have one already
        if not self.instruction_list:
            self.make_tree()

        # ####### #
        # BLOSSOM #
        # ####### #

        # We have two conditions for halting; either a vertex is in danger
        # of having it's weight spread past the time boundary, or we run
        # out of vertices to pair (yay). This loop continues until one of
        # these is reached.

        while True:

            # Get next instruction from list
            if self.instruction_list.empty():
                raise ValueError('Queue empty when attempted' +
                                 'to get instruction!')
            ni = self.instruction_list.get()

            # We have to validate the instruction before use.
            # This is done in here; might want to shift?

            #
            # Collapse tree instruction
            #

            if ni[1] == 0:
                # Get edge:
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]
                # check edge is active
                if not edge[5]:
                    continue
                # Grow to instruction if needed
                if ni[0]-self.total_growth != 0:
                    try:
                        self.grow(ni[0]-self.total_growth)
                    except ValueError:
                        # Reinsert instruction
                        self.instruction_list.put(ni)
                        return
                # collapse tree
                self.collapse_tree(ep, vp)
                # Check if we can make a new tree, otherwise return
                # to user
                if len(self.unpaired_list) == 0:
                    return
                self.make_tree()

            #
            # Add to tree instruction
            #

            elif ni[1] == 1:  # Add to tree instruction
                # This instruction is completed as long as the edge is active

                # Get edge data
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]

                # Check edge is active:
                if not edge[5]:
                    continue

                # If this node is in onl, we don't perform the
                # instruction (analogous to marking the edge)
                # Checking whether this node is in onl is for degenerate
                # edge cases; it could be removed if we prioritised making
                # blossoms over connecting edges, but best to be careful.
                if edge[vp+3] in self.outer_node_list+self.inner_node_list:
                    continue

                # It is possible for a blossom to have an instruction added,
                # then become an inner blossom separately, then break, and
                # the released node that is no longer in the tree would have
                # a different weight than that calculated. So, we need to
                # do a weight check.
                # This check actually catches a *lot* of edge cases, we
                # should consider just doing it at the start.
                if (np.abs(ni[0] - edge[0] + self.node_weight_list[edge[3]]
                    + self.node_weight_list[edge[4]] - self.total_growth)
                    > self._err):
                    continue

                # Grow to instruction if needed
                if ni[0]-self.total_growth != 0:
                    try:
                        self.grow(ni[0]-self.total_growth)
                    except ValueError:
                        self.instruction_list.put(ni)
                        return

                # Add to tree
                self.add_to_tree(ep, vp)

            #
            # Make blossom instruction
            #

            elif ni[1] == 2:
                # This instruction is completed as long as the
                # edge is active
                # Get edge
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]
                # Check edge is active
                if not edge[5]:
                    continue
                # Grow to instruction if needed
                if ni[0]-self.total_growth != 0:
                    try:
                        self.grow(ni[0]-self.total_growth)
                    except ValueError:
                        self.instruction_list.put(ni)
                        return
                # Make blossom!
                self.make_blossom(ep, vp)

            #
            # Break blossom instruction
            #

            elif ni[1] == 3:
                # This should only be performed if our blossom
                # is not inside another blossom
                if not self.blossom_list[ni[2]]:
                    # Grow to instruction if needed
                    if ni[0]-self.total_growth != 0:
                        try:
                            self.grow(ni[0]-self.total_growth)
                        except ValueError:
                            self.instruction_list.put(ni)
                            return
                    self.break_blossom(ni[2])

            else:  # Something is wrong
                raise ValueError('Instruction not understood.')

    def finish(self, boundary_list=None, weight_lists=[], c_flag=False):
        '''
        Adds time boundary vertex to blossom, finishes algorithm,
        returns result.

        Input:
        boundary_list: list of connections from last few
            rows to the time boundary.
        c_flag: flag to save current state and revert afterwards
            as with theoretical input we can have our cake and
            eat it too ;)

        Output:
        pairing: the calculated pairing.
        '''

        if c_flag:
            # Store all data for recovery later
            twl_store = list(self.time_weight_list)
            prl_store = list(self.pairing_real_list)
            ril_store = list(self.reverse_index_list)
            fil_store = list(self.forward_index_list)
            wc_store = self.weight_cap
            pl_store = [list(p) for p in self.pairing_list]
            upl_store = deque(self.unpaired_list)
            nwl_store = list(self.node_weight_list)
            nel_store = [list(n) for n in self.node_edge_list]
            el_store = [list(edges) for edges in self.edge_list]
            bl_store = list(self.blossom_list)
            cl_store = [list(cycle) for cycle in self.cycle_list]
            rl_store = list(self.root_list)
            cel_store = [[list(e) for e in c] for c in self.cycle_edge_list]

            if self.instruction_list is not None:
                il1 = queue.PriorityQueue()
                il_store = queue.PriorityQueue()
                while not self.instruction_list.empty():
                    inst = self.instruction_list.get()
                    il1.put(inst)
                    il_store.put(inst)
                self.instruction_list = il1
            else:
                il_store = None

            tg_store = self.total_growth
            tbl_store = [list(t) for t in self.tree_branch_list]
            onl_store = list(self.outer_node_list)
            inl_store = list(self.inner_node_list)

        if not self.instruction_list and not self.unpaired_list and (weight_lists is None or len(weight_lists) == 0):  # If we have already finished
            # We do not update pairing within the blossoms until the
            # end, as it's too hard to update, but well-defined by the
            # root and cycle. So, before we return to the user,
            # this is all done here.
            self.final_pairing_update()

            if c_flag:
                # Revert back to previous state - not needed on FPGA
                prl_return = self.pairing_real_list
                self.time_weight_list = twl_store
                self.pairing_real_list = prl_store
                self.reverse_index_list = ril_store
                self.forward_index_list = fil_store
                self.weight_cap = wc_store
                self.pairing_list = pl_store
                self.unpaired_list = upl_store
                self.node_weight_list = nwl_store
                self.node_edge_list = nel_store
                self.edge_list = el_store
                self.blossom_list = bl_store
                self.cycle_list = cl_store
                self.root_list = rl_store
                self.cycle_edge_list = cel_store
                self.instruction_list = il_store
                self.total_growth = tg_store
                self.tree_branch_list = tbl_store
                self.outer_node_list = onl_store
                self.inner_node_list = inl_store
                self.time_boundary = None

                if boundary_list is not None:  # If we are expecting a time boundary
                    return prl_return+[None]  # Return our completed pairing
                else:
                    return prl_return

            else:
                if boundary_list is not None:  # If we are expecting a time boundary
                    return self.pairing_real_list+[None]  # Return our completed pairing
                else:
                    return self.pairing_real_list

        if boundary_list is not None:
            self.add_vertex(boundary_list)
            self.time_boundary = self.unpaired_list.pop()  # Pop location of time boundary and store
        else:
            for weight_list in weight_lists:
                self.add_vertex(weight_list)

        # ####### #
        # BLOSSOM #
        # ####### #

        # Make a new tree if we don't have one already
        if not self.instruction_list:
            self.make_tree()

        while True:

            # Get next instruction from list
            if self.instruction_list.empty():
                raise ValueError('Queue empty when attempted to get instruction!')
            ni = self.instruction_list.get()

            # We have to validate the instruction before use.
            # This is done in here; might want to shift?

            #
            # Collapse tree instruction
            #

            if ni[1] == 0:
                # Get edge:
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]
                # check edge is active
                if not edge[5]:
                    continue
                # Grow to instruction if needed
                if ni[0]-self.total_growth != 0:
                    self.grow_no_tb(ni[0]-self.total_growth)
                # collapse tree
                self.collapse_tree(ep, vp)
                # Check if we can make a new tree, otherwise return
                # to user
                if len(self.unpaired_list) == 0:
                    break
                self.make_tree()

            #
            # Add to tree instruction
            #

            elif ni[1] == 1:
                # This instruction is completed as long as the
                # edge is active and the other node is not in inl
                # Get edge
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]
                # Check edge is active:
                if not edge[5]:
                    continue
                # If the other node is in onl or inl, we don't perform the
                # instruction (analogous to marking the edge)
                # Checking whether this node is in onl is for degenerate
                # edge cases; it could be removed if we prioritised making
                # blossoms over connecting edges, but best to be careful.
                if edge[vp+3] in self.inner_node_list+self.outer_node_list:
                    continue
                # It is possible for a blossom to have an instruction added,
                # then become an inner blossom separately, then break, and
                # the released node that is no longer in the tree would have
                # a different weight than that calculated. So, we need to
                # do a weight check.
                # This check actually catches a *lot* of edge cases, we
                # should consider just doing it at the start.
                if (np.abs(ni[0] - edge[0] + self.node_weight_list[edge[3]] + 
                    self.node_weight_list[edge[4]] - self.total_growth) > self._err):
                    continue
                #
                # Grow to instruction if needed
                #
                if ni[0]-self.total_growth != 0:
                    self.grow_no_tb(ni[0]-self.total_growth)
                # Add to tree
                self.add_to_tree(ep, vp)

            #
            # Make blossom instruction
            #

            elif ni[1] == 2:
                # This instruction is completed as long as the
                # edge is active
                # Get edge
                ep = ni[2]
                vp = ni[3]
                edge = self.edge_list[ep]
                # Check edge is active
                if not edge[5]:
                    continue
                if ni[0]-self.total_growth != 0:
                    self.grow_no_tb(ni[0]-self.total_growth)
                # Make blossom!
                self.make_blossom(ep, vp)

            #
            # Break blossom instruction
            #

            elif ni[1] == 3:

                # This should only be performed if our blossom
                # is not inside another blossom
                if not self.blossom_list[ni[2]]:
                    # Grow to instruction if needed
                    if ni[0]-self.total_growth != 0:
                        self.grow_no_tb(ni[0]-self.total_growth)
                    # Execute
                    self.break_blossom(ni[2])

            else:  # Something is wrong
                raise ValueError('Instruction not understood.')

        # We do not update pairing within the blossoms until the
        # end, as it's too hard to update, but well-defined by the
        # root and cycle. So, before we return to the user,
        # this is all done here.
        self.final_pairing_update()

        #
        # Data reversion and result returning
        #

        if c_flag:
            # Revert back to previous state - not needed on FPGA
            prl_return = self.pairing_real_list
            self.time_weight_list = twl_store
            self.pairing_real_list = prl_store
            self.reverse_index_list = ril_store
            self.forward_index_list = fil_store
            self.weight_cap = wc_store
            self.pairing_list = pl_store
            self.unpaired_list = upl_store
            self.node_weight_list = nwl_store
            self.node_edge_list = nel_store
            self.edge_list = el_store
            self.blossom_list = bl_store
            self.cycle_list = cl_store
            self.root_list = rl_store
            self.cycle_edge_list = cel_store
            self.instruction_list = il_store
            self.total_growth = tg_store
            self.tree_branch_list = tbl_store
            self.outer_node_list = onl_store
            self.inner_node_list = inl_store
            self.time_boundary = None

            return prl_return

        else:
            return self.pairing_real_list

    def add_vertex(self, weight_list):
        """
        Adds a vertex to the graph

        Input:
        weight_list: list in the form (index, weight), with each
            term corresponding to an edge we wish to create.

        Output: none

        Currently updates:
        gi, ei, nei, pl, prl, fil, ril, upl, nwl, twl, el,

        Currently uses:
        tbw, bl, onl, wc, tg

        Currently does not touch:
        twl, nri, vb, ul, wc, dil,
        pl, upl, nwl, nel, el, bl, rl, cl, cel, il, tg, tbl, onl, inl

        Currently needs functions:
        None
        """

        # Find graph (gi) and error (ei) index for vertex
        # Whilst removing graph index from the unused list and
        # updating the next error index
        gi = len(self.node_weight_list)
        ei = len(self.pairing_real_list)

        # Update pairing lists (new vertex is unpaired)
        self.pairing_list.append([])
        self.pairing_real_list.append(None)

        # Update reverse and forward index lists
        self.forward_index_list.append(gi)
        self.reverse_index_list.append(ei)

        # Update unpaired list
        self.unpaired_list.append(gi)

        # Initialise current node weight to 0
        self.node_weight_list.append(0)

        # Set the time boundary to one time constant
        self.time_weight_list.append(self.time_boundary_weight)

        # Add elements to other lists
        self.blossom_list.append(None)
        self.root_list.append(None)
        self.cycle_list.append([])
        self.cycle_edge_list.append([])
        self.tree_branch_list.append([])

        # Edge insertion

        # Run over all input edges
        self.node_edge_list.append([])

        edge_index = len(self.edge_list)

        for data_set in weight_list:

            # graph index of other vertex
            gi2 = self.forward_index_list[data_set[0]]

            # get v1 and v2 for the edge, these are also currently
            # n1 and n2
            if gi < gi2:
                v1 = gi
                v2 = gi2
                n1 = gi
                n2 = gi2
                vp = 1
                # Store result for later
                less_than_flag = True
            else:
                v1 = gi2
                v2 = gi
                n1 = gi2
                n2 = gi
                vp = 0
                # Store result for later
                less_than_flag = False

            # insert the edge into the current node edge lists
            self.node_edge_list[gi].append([edge_index, vp])
            self.node_edge_list[gi2].append([edge_index, 1-vp])

            # Flag whether we encounter another edge from a blossom
            # to the same vertex
            other_edge_flag = False

            # get weight, and go up through blossoms till
            # the top, each time subtracting the node weight
            # *** of the previous blossom ***
            weight = data_set[1]

            while self.blossom_list[gi2]:  # While we are inside another blossom

                # move up to the new blossom
                oldgi2 = gi2
                gi2 = self.blossom_list[gi2]

                # check if we already have an edge between these two
                # if so, and it is lower weight, we can stop here.
                # if so, and it is higher weight, we update that weight
                # and move on.

                # No need to search - if we have added an edge it must
                # be the last pointer on the edge list of the other node!
                # this can be split into the edge pointer and the vertex
                # pointer (refer to definition of nel)
                ep, vp_to_check = self.node_edge_list[gi2][-1]

                # Get edge from edge list
                edge_to_check = self.edge_list[ep]

                # Check if it connects to gi
                # vp+1 is the vertex that gi2 connects to through the edge
                if edge_to_check[vp_to_check+1] == gi:

                    weight_to_check = edge_to_check[0]
                    temp_gi = gi2
                    while temp_gi is not None and temp_gi != edge_to_check[4-vp_to_check]:
                        weight_to_check += self.node_weight_list[temp_gi]
                        temp_gi = self.blossom_list[temp_gi]


                    # If our current edge has been beaten
                    if weight_to_check <= weight - self.node_weight_list[oldgi2]:

                        # We still need to make the edge
                        edge = [weight, v1, v2, n1, n2, False]

                        # And add it to the list
                        self.edge_list.append(edge)
                        break

                    else:
                        # Flag that we now have to change the last entry
                        # of the edge list for all nodes above this.
                        other_edge_flag = True

                        # We need to bring the node index for the other edge
                        # back down to the node before this.
                        #
                        # 2-vp is the vertex inside gi2 that this
                        # connects to.
                        #
                        # Note that this can be done in parallel
                        # to other operations
                        n2_new = edge_to_check[2-vp_to_check]

                        # we shift up the blossom list until we can
                        # see the node we are currently at
                        while self.blossom_list[n2_new] != gi2:
                            n2_new = self.blossom_list[n2_new]

                        # and this becomes the new node.
                        # 4-vp points to the position of the node in the edge
                        edge_to_check[4-vp_to_check] = n2_new

                        edge_to_check[0] += self.node_weight_list[n2_new]

                        # flag edge as inactive
                        edge_to_check[5] = False

                # If we have made it this far, we can increase our node by one
                # and decrement our weight
                if less_than_flag:
                    weight -= self.node_weight_list[n2]
                    n2 = gi2
                else:
                    weight -= self.node_weight_list[n1]
                    n1 = gi2

                # Add edge to list of edges on node. How we do this depends on
                # the other_edge flag, as if it is True, we already have
                # such an edge, and we just need to change the index.

                if other_edge_flag:
                    self.node_edge_list[gi2][-1] = [edge_index, 1-vp]
                else:
                    self.node_edge_list[gi2].append([edge_index, 1-vp])

            # If we are at an active vertex that is an outer node in the tree,
            # we need to update the instruction list.

            # The else executes if the while loop exits because the condition
            # became false, which implies we made it to the top of the blossom.
            else:

                edge = [weight, v1, v2, n1, n2, True]
                self.edge_list.append(edge)

                # Need to check we currently have a tree, and then that
                # the other node is in the outer node list.
                if self.instruction_list and gi2 in self.outer_node_list:
                    self.make_instructions([[edge_index, 1-vp]])

            # Check that the inserted edge is not over-tight, if so, reduce
            # weight on other node to 0, break any associated blossoms,
            # remove any edges and destroy any required trees
            if weight - self.node_weight_list[gi2] < 0:

                # Shift to active other node (i.e through blossoms)
                while self.blossom_list[gi2]:
                    gi2 = self.blossom_list[gi2]

                # The other node might not be in the unpaired_list now, so
                # we need to check and add it
                if gi2 not in self.unpaired_list:
                    self.unpaired_list.appendleft(gi2)

                # If we have a tree its instruction list is corrupted
                # by the random pairing/unpairing of vertices outside.
                # Fair easier to destroy the tree and start again

                # Check that a tree exists
                if self.instruction_list:

                    # Add the tree root back to the list of unpaired nodes
                    # so that we attempt to remake this tree
                    if self.outer_node_list[0] != gi2:
                        self.unpaired_list.appendleft(self.outer_node_list[0])

                    # Remove the instruction list defining the tree,
                    # which flags to the rest of the program that there
                    # is none.
                    self.instruction_list = None

                # Next, we have to go through blossoms from the top down,
                # at each point breaking whatever they are matched to,
                # re-adding the other vertex to the unpaired vertex list,
                # and then breaking the blossom.

                # Re-obtain the root vertex
                vertex_to_free = self.forward_index_list[data_set[0]]

                while True:

                    # Start at the vertex
                    this_node = vertex_to_free

                    # Shift to active node (i.e through blossoms)
                    while self.blossom_list[this_node]:
                        this_node = self.blossom_list[this_node]

                    # If this node is paired, unpair it
                    if self.pairing_list[this_node]:
                        other_node = self.pairing_list[this_node][0]

                        # Delete pairing
                        self.pairing_list[this_node] = []
                        self.pairing_list[other_node] = []

                        # Put nodes on unpaired list
                        if this_node not in self.unpaired_list:
                            self.unpaired_list.appendleft(this_node)
                        if other_node != 0 and (not self.time_boundary or other_node != self.time_boundary):
                            self.unpaired_list.appendleft(other_node)

                    # Reduce this vertex weight to 0
                    self.node_weight_list[this_node] = 0

                    # If our edge is no longer over-tight, we are done.
                    if edge[0] > 0:

                        break

                    # Otherwise, we must be in a blossom, which we break.
                    self.break_blossom_no_tree(this_node)

            # increase the edge index for the next edge
            edge_index += 1

    '''
    #######################################################################
    Below here are the mid-level algorithms; algorithms that are present in the
    blossom flowchart
    #######################################################################
    '''

    def make_tree(self):
        '''
        Create new alternating tree taking the first vertex from the
        list of unpaired vertices.

        Note that it is impossible to start a tree with an unpaired root vertex
        inside a blossom. This would imply that this vertex was inside a
        previous tree (which formed the blossom), but every vertex that has
        been inside a previous tree is matched.
        '''

        # Get tree root as the first vertex to be added that is not yet paired.
        root = self.unpaired_list.popleft()

        # We might want to consider a quick check of the most likely outcome
        # i.e. that root pairs to another unpaired vertex immediately
        # as this will save us the trouble of setting up the tree etc.

        # The root has no branch back, so we remove any possible edge from a
        # previous tree.
        self.tree_branch_list[root] = []

        # Our inner node list is empty,
        # but our outer node list contains the root
        self.inner_node_list = []
        self.outer_node_list = [root]

        # Reset our weight_cap
        self.weight_cap = None  # Technically infinity

        # Reset our instruction list
        self.instruction_list = queue.PriorityQueue()

        # Reset total growth
        self.total_growth = 0

        self.make_instructions(self.node_edge_list[root])

    def grow(self, delta_weight):
        """
        Increase the weight of the outer nodes by delta_weight and decrease the
        weight of all inner nodes by delta_weight.

        Reverts and throws error if we hit the time boundary

        @param delta_weight: the weight to be increased or decreased.
        """

        increased_list = []
        # Run through our outer node list
        for index in self.outer_node_list:

            # Check if our weight is pushing past the time boundary
            # as then we need to halt. As our algorithm has some time
            # after this to sit around and twiddle its thumbs, we do
            # this check as we run through the list, rather than beforehand
            # and suck up the cost of having to undo things.
            if self.node_weight_list[index] + delta_weight > self.time_weight_list[index]:

                # Revert everything;
                for index_new in increased_list:

                    # Reduce weight
                    self.node_weight_list[index_new] -= delta_weight

                # Raise error; this is handled by the run_blossom algorithm
                raise ValueError('Attempted to grow past time boundary')

            # Otherwise, we're free to increment the weight!

            # Increase outer node weight
            self.node_weight_list[index] += delta_weight
            increased_list.append(index)

        # If we have gotten to this point we're safe; we cant hit the
        # time boundary if we're decreasing weights.
        for index in self.inner_node_list:
            self.node_weight_list[index] -= delta_weight

        # increment the total growth
        self.total_growth += delta_weight

    def grow_no_tb(self, delta_weight):
        '''
        Increase the weight of the outer nodes by delta_weight and decrease the
        weight of all inner nodes by delta_weight.

        Does not check for time boundary - appropriate only for final round

        @param delta_weight: the weight to be increased or decreased.
        '''
        # Increment outer node weights
        for index in self.outer_node_list:
            self.node_weight_list[index] += delta_weight

        # Decrement inner node weights
        for index in self.inner_node_list:
            self.node_weight_list[index] -= delta_weight

        # Increment total growth
        self.total_growth += delta_weight

    def break_blossom(self, v):
        """
        Expand a blossom, removing all data about it from graph.

        @param v: the blossom to be expanded

        Currently updates:
        wc*, dil, pl*, prl*, nel, el, bl, rl, cl, cel, il*, tbl,
        onl, inl

        Currently uses:
        tg*

        Currently does not touch:
        tbw, twl, tb, ril, fil, nei, nri, vb, ul, dil,
        upl, nwl, il, tg,

        Currently needs functions:
        make_instructions
        pair
        """

        # Get cycle and root
        cycle = self.cycle_list[v]
        root = self.root_list[v]
        cycle_edge_list = self.cycle_edge_list[v]

        # Quick fix to pairing list so that our root is now paired
        # to whatever our blossom is paired to - note that our
        # blossom must be paired to a vertex that is not the root
        # or the time boundary!
        paired_node, ep = self.pairing_list[v]
        self.pairing_list[root] = [paired_node, ep]
        self.pairing_list[paired_node] = [root, ep]

        # Also need to update the tree for the root and fix the inl
        self.tree_branch_list[paired_node][0] = root
        self.inner_node_list.remove(v)

        # Get edge connecting back in the tree, and find where
        # this connects to in the cycle
        v_out_connected, ep, vp = self.tree_branch_list[v]
        edge = self.edge_list[ep]
        # Go down to the bottom vertex
        v_out = edge[2-vp]
        # Step up through blossoms until we are one node below
        while self.blossom_list[v_out] != v:
            v_out = self.blossom_list[v_out]

        # Get indices in cycle - unfortunately this has to be done
        # via a search
        out_index = cycle.index(v_out)
        root_index = cycle.index(root)

        # Begin list of edges to make instructions from. We don't put them in
        # till the end, due to some edge cases that make bad instructions.
        # Specifically, if an vertex is never paired until it enters the
        # blossom we can have a situation where a 'collapse tree' instruction
        # is erroneously made.

        edges_for_instructions = []

        # We want to loop over cycle, but we need to do something slightly
        # different depending on whether we have the root, the out_node,
        # or a node that will become inner, or outer, or not either.
        # This gives the following a lot of cases to consider.

        # Need to make a direction flag to figure out which side of
        # out_index the nodes inside the tree lie.
        # This also tells us which way the index should be offset
        # when looking for the edges
        outside = bool((out_index-root_index) % 2)
        if (out_index > root_index) ^ outside == 0:
            direction = 1
            index_offset = False
        else:
            direction = -1
            index_offset = True

        # The out index node gets its tbl from a different place
        # and so should be treated differently
        self.tree_branch_list[v_out] = list(self.tree_branch_list[v])
        self.blossom_list[v_out] = None
        self.inner_node_list.append(v_out)
        # If blossom, insert break instruction into il
        if self.cycle_list[v_out] and self.node_weight_list[v_out] + self.total_growth < self.weight_cap+self._err:
            inst = [self.node_weight_list[v_out]+self.total_growth, 3, v_out]
            self.instruction_list.put(inst)
        node_weight = self.node_weight_list[v_out]
        for ep, vp in self.node_edge_list[v_out]:
            # Need to update effective weights for
            # edges that were connected to the blossom
            if self.edge_list[ep][4-vp] != v_out:
                # Update effective weight
                self.edge_list[ep][0] += node_weight
                # Pull back node to current
                self.edge_list[ep][4-vp] = v_out
            # If the other side of the edge is in a blossom
            # The edge remains inactive
            if not self.blossom_list[self.edge_list[ep][3+vp]]:
                self.edge_list[ep][5] = True

        #
        # Inner nodes
        #
        index = out_index
        # Cycle through until we get to the root
        while index != root_index:

            # Next index, looping circularly
            index += 2*direction
            if index < 0:
                index += len(cycle)
            if index >= len(cycle):
                index -= len(cycle)

            # Get node index
            ni = cycle[index]
            # Insert into inner node list
            self.inner_node_list.append(ni)
            # If blossom, insert break instruction into il
            if self.cycle_list[ni] and self.node_weight_list[ni] + self.total_growth < self.weight_cap+self._err:
                inst = [self.node_weight_list[ni]+self.total_growth, 3, ni]
                self.instruction_list.put(inst)
            # Insert appropriate edge into tbl
            if index_offset:
                new_ep, new_vp = cycle_edge_list[index]
                self.tree_branch_list[ni] = [self.edge_list[new_ep][3+new_vp], new_ep, new_vp]
            else:
                new_ep, new_vp = cycle_edge_list[index-1]
                self.tree_branch_list[ni] = [self.edge_list[new_ep][4-new_vp], new_ep, 1-new_vp]
            # Update blossom list
            self.blossom_list[ni] = None
            # Get node weight
            node_weight = self.node_weight_list[ni]
            # Loop over all edges attached to node
            for ep, vp in self.node_edge_list[ni]:
                # Need to update effective weights for
                # edges that were connected to the blossom
                if self.edge_list[ep][4-vp] != ni:
                    # Update effective weight
                    self.edge_list[ep][0] += node_weight
                    # Pull back node to current
                    self.edge_list[ep][4-vp] = ni
                # If the other side of the edge is in a blossom
                # The edge remains inactive
                if not self.blossom_list[self.edge_list[ep][3+vp]]:
                    self.edge_list[ep][5] = True

        #
        # Neither nodes
        #

        index = out_index - direction
        if index < 0:
            index += len(cycle)
        elif index >= len(cycle):
            index -= len(cycle)
        make_edge = True
        # Cycle through until we get to the root
        while index != root_index:

            # Get node index
            ni = cycle[index]
            # Update blossom list
            self.blossom_list[ni] = None
            # Get node weight
            node_weight = self.node_weight_list[ni]
            # Start set of edges to make instructions for.
            # This is the list of active edges attached to
            # outer edges.
            edge_to_outer_list = []
            # Loop over all edges attached to node
            for ep, vp in self.node_edge_list[ni]:
                # Need to update effective weights for
                # edges that were connected to the blossom
                if self.edge_list[ep][4-vp] != ni:
                    # Update effective weight
                    self.edge_list[ep][0] += node_weight
                    # Pull back node to current
                    self.edge_list[ep][4-vp] = ni
                # If the other side of the edge is in a blossom
                # The edge remains inactive
                if self.blossom_list[self.edge_list[ep][3+vp]]:
                    continue
                self.edge_list[ep][5] = True
                # Add edge to instruction list if we're in the onl
                if self.edge_list[ep][3+vp] in self.outer_node_list:
                    # The instruction should be added as if
                    # it was coming from the other side, so
                    # switch vp.
                    edge_to_outer_list.append([ep, 1-vp])

            # If we are an odd number of edges
            # from the root, pair
            if make_edge:
                if index_offset:
                    self.pair(cycle_edge_list[index][0])
                else:
                    self.pair(cycle_edge_list[index-1][0])
            # Flip for the next
            make_edge = not make_edge

            # Add instructions to list
            # All of these instructions are the same type
            # but I'm not sure we can get much by using this
            # fact above just going to the usual function
            edges_for_instructions += edge_to_outer_list

            # Increment index
            index -= direction
            if index < 0:
                index += len(cycle)
            elif index >= len(cycle):
                index -= len(cycle)

        #
        # Outer nodes
        #

        # Run over new outer nodes, updating their blossom index, tbl, and
        # reactivating edges and adding new instructions

        index = out_index + direction
        if index < 0:
            index += len(cycle)
        if index >= len(cycle):
            index -= len(cycle)
        while index != (root_index + direction) % len(cycle):

            # Get node index
            ni = cycle[index]
            # Insert into outer node list
            self.outer_node_list.append(ni)
            # Insert appropriate edge into tbl
            if index_offset:
                new_ep, new_vp = cycle_edge_list[index]
                self.tree_branch_list[ni] = [self.edge_list[new_ep][3+new_vp], new_ep, new_vp]
            else:
                new_ep, new_vp = cycle_edge_list[index-1]
                self.tree_branch_list[ni] = [self.edge_list[new_ep][4-new_vp], new_ep, 1-new_vp]
            # Update blossom list
            self.blossom_list[ni] = None
            # Get node weight
            node_weight = self.node_weight_list[ni]
            # Loop over all edges attached to node
            for ep, vp in self.node_edge_list[ni]:
                # Need to update effective weights for
                # edges that were connected to the blossom
                if self.edge_list[ep][4-vp] != ni:
                    # Update effective weight
                    self.edge_list[ep][0] += node_weight
                    # Pull back node to current
                    self.edge_list[ep][4-vp] = ni
                # If the other side of the edge is in a blossom
                # The edge remains inactive
                if not self.blossom_list[self.edge_list[ep][3+vp]]:
                    self.edge_list[ep][5] = True
            edges_for_instructions += self.node_edge_list[ni]
            # These edges always pair
            self.pair(new_ep)
            # Next index, looping circularly
            index += 2*direction
            if index < 0:
                index += len(cycle)
            if index >= len(cycle):
                index -= len(cycle)

        # Make all instructions simultaneously.
        self.make_instructions(edges_for_instructions)

        # Delete cycle list, removing the record of this
        # being a blossom
        self.cycle_list[v] = []

    def break_blossom_no_tree(self, v):
        """
        Expand a blossom, removing all data about it from graph.
        This function is for when we are not within a tree
        and do not need to update any tree data.

        @param v: the blossom to be expanded
        """

        # Get cycle
        cycle = self.cycle_list[v]
        # Check that we haven't done something terribly wrong
        if cycle is None:
            raise ValueError('This is no blossom, this is a critical error!' +
                             ' Perhaps you put in a negatively weighted edge?')

        # Get other parts of blossom
        root = self.root_list[v]
        cycle_edge_list = self.cycle_edge_list[v]

        # Quick fix to pairing list so that our root is now paired
        # to whatever our blossom is paired to
        if self.pairing_list[v]:
            paired_node, ep = self.pairing_list[v]
            self.pairing_list[root] = [paired_node, ep]

            # Note that we do not pair the boundary to anything!
            if paired_node != 0:
                self.pairing_list[paired_node] = [root, ep]
        else:
            # If we have no pairing, this vertex will
            # be in the unpaired list, and needs to be
            # replaced by its root. This requires a search
            upl_index = self.unpaired_list.index(v)
            self.unpaired_list[upl_index] = self.root_list[v]
            self.pairing_list[self.root_list[v]] = []

        root_index = cycle.index(root)

        # Loop over other nodes in the cycle, pairing them
        # to the root and restoring their edges

        index = root_index

        make_edge = False

        # Cycle through until we get to the root
        while True:

            # Get node index
            ni = cycle[index]
            # Update blossom list
            self.blossom_list[ni] = None
            # Get node weight
            node_weight = self.node_weight_list[ni]

            # Loop over all edges attached to node
            for ep, vp in self.node_edge_list[ni]:
                # Need to update effective weights for
                # edges that were connected to the blossom
                if self.edge_list[ep][4-vp] != ni:
                    # Update effective weight
                    self.edge_list[ep][0] += node_weight
                    # Pull back node to current
                    self.edge_list[ep][4-vp] = ni
                # If the other side of the edge is in a blossom
                # The edge remains inactive
                if self.blossom_list[self.edge_list[ep][3+vp]]:
                    continue
                self.edge_list[ep][5] = True

            # If we are an odd number of edges
            # from the root, pair
            if make_edge:
                self.pair(cycle_edge_list[index][0])
            # Flip for the next
            make_edge = not make_edge

            # Increment index
            index += 1
            if index >= len(cycle):
                index -= len(cycle)

            # If we have gotten back to the root, we are finished.
            if index == root_index:
                break

        # Delete cycle list, removing the record of this
        # being a blossom
        self.cycle_list[v] = []

    def add_to_tree(self, ep, vp):
        """
        Add a node to an existing node in the tree.
        T: also need to add the partner

        @param parent_node: the node which already exists in the tree.
        @param child_node: the edge index in el[parent_node] that we
            want to connect to

        If parent_node does not exist in the tree, raise an error.

        Currently needs functions:
        make_instructions
        """

        # Get edge, new node, and old node
        edge = self.edge_list[ep]

        new_inner_node = edge[3+vp]
        old_node = edge[4-vp]

        # add new node to inner node list
        self.inner_node_list.append(new_inner_node)

        # add edge to tree branch list
        self.tree_branch_list[new_inner_node] = [old_node, ep, 1-vp]

        # get new outer node, add to outer node list
        new_outer_node, ep_new = self.pairing_list[new_inner_node]
        self.outer_node_list.append(new_outer_node)

        # add edge back to tree branch list, calculating the vp
        # on the spot
        if self.edge_list[ep_new][3] == new_outer_node:
            self.tree_branch_list[new_outer_node] = [new_inner_node, ep_new, 1]
        else:
            self.tree_branch_list[new_outer_node] = [new_inner_node, ep_new, 0]

        # Update instruction list

        # If the inner node is a blossom, need to add the break instruction
        # Check via presence of a cycle
        if self.cycle_list[new_inner_node]:

            # Make instruction and insert
            inst = [self.node_weight_list[new_inner_node]+self.total_growth, 3, new_inner_node]
            self.instruction_list.put(inst)

        self.make_instructions(self.node_edge_list[new_outer_node])

    def collapse_tree(self, ep, vp):
        """
        Find path from tree back to root, update pairing, destroy tree.

        @param end_node: the vertex at the end of the tree.
        @param pair_vertex: the (previously) unpaired vertex end_vertex will be
                paired to. Note that this *must* be a vertex rather than a
                blossom, as it is outside the tree, and any blossom that is
                outside the tree must be paired.
        @param pair_edge: the index of the connecting edge in self.edge_list[end_node]

        Currently updates:
        upl, il

        Currently uses:
        el, cl, tbl, dil

        Currently does not touch:
        tbw, twl, tb, nei, pl, prl, ril, fil, ul, wc, nwl, bl, cel, tg,
        onl, inl, rl, nel, vb, nri

        Currently needs functions:
        new_root
        recycle

        """

        # Extract final edge and thus the two vertices
        edge = self.edge_list[ep]

        # Vertices are labelled as either odd_v or even_v depending
        # on the parity of the number of edges from the root.
        # So, the root is an even_v, the vertex we are connecting
        # to is an odd_v, and the vertex we are connecting from
        # is an even_v. We go through pairs of even_v and odd_v,
        # pairing each.

        # If we enter an even_v blossom, the entrance point is now
        # the new root. If we enter an odd_v blossom, the entrance
        # point *was* the root, and the new root is the vertex
        # that connects to the previous node outside the blossom.

        odd_v = edge[3+vp]
        even_v = edge[4-vp]

        # Remove odd_v from the unpaired vertex list if there.
        if not self.pairing_list[odd_v] and \
           odd_v != 0 and (not self.time_boundary or
                           odd_v != self.time_boundary):
            self.unpaired_list.remove(odd_v)

        # note that odd_v cannot be a blossom!

        # Go through all nodes in branch till we hit the root.
        # This is flagged by even_v not having a branch back
        while self.tree_branch_list[even_v]:

            # pair the nodes via the edge
            self.pair(ep)

            # and step back twice
            odd_v = self.tree_branch_list[even_v][0]
            even_v, ep, vp = self.tree_branch_list[odd_v]

        # Pair to root:
        self.pair(ep)

        # We check the presence of a tree by its instruction list,
        # so this needs to be removed. The rest of the data is now
        # garbage and will be written over when the new tree comes in.
        self.instruction_list = None

    def make_blossom(self, ep, vp):
        '''
        Create a blossom by tracing v1 and v2 back along the tree to find a
        common root, then generating the odd cycle.

        @params v1, v2: ends of the tree that will join to make a blossom.

        Currently updates:
        twl, el, ul, wc, nwl, nel, bl, rl, cl, cel, il, tbl, onl

        Currently uses:
        pl, tg

        Currently does not touch:
        tbw, tb, prl, ril, fil, nei, nri, vb, dil, upl, tbl, inl

        Currently needs functions:
        get_cycle
        make_node_edge
        make_instructions
        '''

        # Get edge and thus both end nodes
        edge = self.edge_list[ep]
        n1, n2 = edge[3+vp], edge[4-vp]

        # Get the cycle and edge list for the blossom
        cycle, edge_list = self.get_cycle(n1, n2, ep, vp)

        # Get index of new blossom
        nbi = len(self.node_weight_list)  # nbi = 'new blossom index'

        # If our blossom is paired, update the pairing in pl. This is
        # a slightly delicate operation and unique to this case; we don't need
        # to update prl, and our edge isn't updated, so we do it here rather
        # than sending to pair.
        if self.pairing_list[cycle[0]]:
            vp, ep = self.pairing_list[cycle[0]]
            self.pairing_list.append([vp, ep])
            if vp != 0 and vp != self.time_boundary:
                self.pairing_list[vp][0] = nbi
        else:
            self.pairing_list.append([])

        # Insert cycle and edge list
        self.cycle_list.append(cycle)
        self.cycle_edge_list.append(edge_list)
        self.reverse_index_list.append(None)

        # Our root is currently the first entry in cycle by construction
        self.root_list.append(cycle[0])
        self.blossom_list.append(None)

        # Set node weight list to zero (previously it may have been garbage)
        self.node_weight_list.append(0)
        self.time_weight_list.append(None)

        # Update the tree branch list
        if self.tree_branch_list[cycle[0]]:  # Unless the cycle is the new root
            # Copy it
            self.tree_branch_list.append(list(self.tree_branch_list[cycle[0]]))
        else:
            self.tree_branch_list.append([])

        # The edges for the other nodes in the tree are still the same
        # but their ends have changed, and we're currently storing these
        # separately.

        # Outer nodes are not paired to the blossom
        for node in self.inner_node_list:
            if self.tree_branch_list[node] and \
              self.tree_branch_list[node][0] in cycle:
                self.tree_branch_list[node][0] = nbi

        # Create the edge list for the tree. This is a horrible process,
        # Need to think about how to streamline it better.
        self.make_node_edge_list(nbi, cycle)

        # Update the tree node lists
        if self.outer_node_list[0] in cycle:
            self.outer_node_list.append(self.outer_node_list[0])
            self.outer_node_list[0] = nbi
        else:
            self.outer_node_list.append(nbi)
        for node in cycle:
            self.blossom_list[node] = nbi
            try:
                self.outer_node_list.remove(node)
            except ValueError:
                self.inner_node_list.remove(node)

        self.make_instructions(self.node_edge_list[nbi])

    def final_pairing_update(self):
        '''
        Currently, we do not update the pairing within any blossoms.
        This means that before we send pairing results to the user,
        we have to do such an update.
        '''
        for node in range(len(self.node_weight_list)):
            if self.cycle_list[node] and not self.blossom_list[node]:
                self.update_prl(node)

###############################################################################
# Bottom level algorithms: algorithms that directly access data/common routines
###############################################################################

    # Data management - functions that deal with the finite memory we
    #                   have access to

    def get_full_cycle(self, node):
        '''
        Gets an unordered cycle of a blossom;
        i.e. the list of all nodes contained within.
        For memory management purposes.
        '''

        # Copy cycle from cycle list
        top_list = list(self.cycle_list[node])
        node_list = list(top_list)
        # Go through cycle; if any nodes are also blossoms
        # we add their cycles to the list also
        for x in top_list:
            if self.cycle_list[x]:
                node_list += self.get_full_cycle(x)

        return node_list

    def update_prl(self, node):
        '''
        Updates the pairing real list for all vertices on the
        interior of a blossom. Assumes node is a blossom.
        Also assumes that the root is already paired, as this
        is currently done in pair.
        '''
        cycle = self.cycle_list[node]
        root = self.root_list[node]
        ri = cycle.index(root)
        edge_list = self.cycle_edge_list[node]

        # Get the parity of root index to find out
        # What indices to run over
        parity = ri % 2

        # Check if the root is also a blossom
        if self.cycle_list[root]:
            self.update_prl(root)

        # Run over indices before ri
        for index in range(parity, ri, 2):
            # Pair edges without making new roots
            self.pair(edge_list[index][0])

            # If either node is another blossom, enter and pair also
            if self.cycle_list[cycle[index]]:
                self.update_prl(cycle[index])
            if self.cycle_list[cycle[index+1]]:
                self.update_prl(cycle[index+1])

        # Run over indices after ri
        for index in range(ri+1, len(cycle), 2):
            # Pair edges without making new roots
            self.pair(edge_list[index][0])

            # If either node is another blossom, enter and pair also
            if self.cycle_list[cycle[index]]:
                self.update_prl(cycle[index])
            if self.cycle_list[cycle[(index+1) % len(cycle)]]:
                self.update_prl(cycle[(index+1) % len(cycle)])

    def pair(self, ep):
        '''
        Pair up two nodes in pl and prl
        If they are blossoms, the prl roots are paired instead.
        '''
        edge = self.edge_list[ep]
        w, v1, v2, n1, n2, active = edge

        if n1 != 0 and (not self.time_boundary or n1 != self.time_boundary):
            self.pairing_list[n1] = [n2, ep]
            self.pairing_real_list[self.reverse_index_list[v1]] = \
                self.reverse_index_list[v2]

        if n2 != 0 and (not self.time_boundary or n2 != self.time_boundary):
            self.pairing_list[n2] = [n1, ep]
            self.pairing_real_list[self.reverse_index_list[v2]] = \
                self.reverse_index_list[v1]

        # reset roots of any blossoms
        if v1 != n1:
            self.new_root(v1)
        if v2 != n2:
            self.new_root(v2)

    def new_root(self, node):
        '''
        Shift the root of a blossom

        Input:
        blossom - index to the node we are changing the root of
        ep - index to the edge in
        vp - pointer to the vertex within the blossom (edge[1+vp])

        Output:
        none
        '''
        while self.blossom_list[node]:
            blossom = self.blossom_list[node]
            self.root_list[blossom] = node
            node = blossom

    def get_cycle(self, n1, n2, ep, vp):
        '''
        Makes a cycle for a blossom, by running back through a tree
        until a common node is found. This node is the new root of
        the tree, and is put at the start of the returned cycle

        Input:
        n1, n2: nodes at the end of the tree to be connected
        edge: the edge connecting them
        '''
        # Trace back paths to common root, storing edges as we go
        p1l = [n1]  # path 1
        e1l = []
        p2l = [n2]  # path 2
        e2l = []

        # This loop continues until a node appears in both lists.
        loop_count = 0
        while True:
            loop_count += 1
            if loop_count > 20:
                raise ValueError('Caught infinite loop')

            # Trace back two at a time in first path
            # Don't trace back if we're at the root
            if self.tree_branch_list[n1]:
                # inner node connected to n1
                in1, ein1, vpi1 = self.tree_branch_list[n1]
                # new outer node connected to in1
                n1, en1, vp1 = self.tree_branch_list[in1]

                # Store pointers
                p1l += [in1, n1]
                e1l += [[ein1, 1-vpi1], [en1, 1-vp1]]

            # Check if break condition achieved
            try:
                # Attempt to find the index of n1 in p2l. If we can do this,
                # we have completed our cycle, and we can concatenate and exit.
                # otherwise, this throws a ValueError, which is caught, and we
                # pass.
                index = p2l.index(n1)

                # Create cycle and edge list. The slice notation is
                # because we need to reverse p1l and e1l.
                cycle = p1l[::-1] + p2l[:index]
                edge_list = e1l[::-1] + [[ep, 1-vp]] + e2l[:index]

                # Break loop
                break

            except ValueError:  # Not finished yet
                pass

            # Trace back two at a time in second path
            # Don't trace back if we're at the root
            if self.tree_branch_list[n2]:
                in2, ein2, vpi2 = self.tree_branch_list[n2]
                n2, en2, vp2 = self.tree_branch_list[in2]

                # Store pointers
                p2l += [in2, n2]
                e2l += [[ein2, vpi2], [en2, vp2]]

            # Check if break condition achieved
            try:
                # Attempt to find the index of n1 in p2l. If we can do this,
                # we have completed our cycle, and we can concatenate and exit.
                # otherwise, this throws a ValueError, which is caught, and we
                # pass.
                index = p1l.index(n2)

                # Create cycle and edge list. The slice notation is
                # because we need to reverse p1l and e1l.
                cycle = p1l[index::-1] + p2l[:-1]

                # Somewhat bad code; we cut one edge off the edge list
                # of e1l (relative to p1l), but if I do this in python
                # with index==0 it thinks I'm going to the back of the list.
                if index == 0:
                    edge_list = [[ep, 1-vp]] + e2l[:]
                else:
                    edge_list = e1l[index-1::-1] + [[ep, 1-vp]] + e2l[:]

                # Break loop
                break

            except ValueError:  # Not finished yet
                pass
        return cycle, edge_list

    def make_node_edge_list(self, nbi, cycle):
        '''Updates the node edge list for a blossom with given cycle and index.

        This is a fairly complicated task, as we have to avoid doubly including
        edges etc. Hence it is split off of the make_blossom function for ease
        of sight. It can be parallelised, but I'm worried that maybe we should
        be doing it in advance anyway.

        Input:
        nbi: new blossom index
        cycle: cycle of the new blossom.

        '''

        # Find lowest weight edges leading from cycle and store,
        # Whilst deactivating as appropriate.
        # This is pretty slow, as we have to check we have found the
        # lowest weight edge. Might want to think about pregenerating
        # this data for instructions before we know if we need it

        # For ease of access, we store the lowest weight for each index and
        # the corresponding index in self.node_edge_list.
        lowest_weight_list = [None for x in range(len(self.node_weight_list))]
        edge_index_list = [None for x in range(len(self.node_weight_list))]

        num_edges = 0

        self.node_edge_list.append([])

        for node in cycle:

            # And update the tree branch list
            self.tree_branch_list[node] = []

            # And find the lowest weight to the time boundary
            check_twl = self.time_weight_list[node]-self.node_weight_list[node]
            if self.time_weight_list[nbi] is None or \
               self.time_weight_list[nbi] > check_twl:

                self.time_weight_list[nbi] = check_twl

            # Run over the edges
            for ep, vp in self.node_edge_list[node]:

                edge = self.edge_list[ep]

                # Get other end of edge
                other_vertex = edge[1+vp]
                other_node = edge[3+vp]

                # Check if node is already in tree
                if other_node in cycle:
                    # Deactivate edge, break
                    edge[5] = False
                    continue
                elif self.blossom_list[other_node]:
                    top_node = self.blossom_list[other_node]
                    while self.blossom_list[top_node]:
                        top_node = self.blossom_list[top_node]
                    if top_node in cycle:
                        edge[5] = False
                        continue
                # See if another edge to this node already exists
                if lowest_weight_list[other_vertex] is not None:

                    # Calculate the true weight of this edge:
                    # this requires subtracting any higher nodes
                    eff_weight = edge[0] - self.node_weight_list[node] -\
                                 self.node_weight_list[other_node]

                    while self.blossom_list[other_node]:
                        other_node = self.blossom_list[other_node]
                        eff_weight -= self.node_weight_list[other_node]

                    # There is a boundary case here if we have two tight
                    # edges pointing to the same inner node, and one of these
                    # is in tbl, we need to ensure this one is chosen.
                    # We check that other_node is in self.inner_node_list
                    # because self.tree_branch_list may be garbage
                    if np.abs(eff_weight - lowest_weight_list[other_vertex]) <\
                       self._err and (other_node in self.inner_node_list) and \
                       self.tree_branch_list[other_node][1] == ep:

                        # Get the index this was inserted at
                        edge_index = edge_index_list[other_vertex]

                        # The previous edge is now inactive
                        # Get pointer to prev. edge
                        pep = self.node_edge_list[nbi][edge_index][0]
                        # Insert
                        self.edge_list[pep][5] = False

                        # Update node edge list with our new edge (no appending needed)
                        self.node_edge_list[nbi][edge_index] = [ep, vp]

                        # Update the weight associated to this node.
                        lowest_weight_list[other_vertex] = eff_weight

                    # Strictly less than
                    elif eff_weight < lowest_weight_list[other_vertex]-self._err:

                        # Get the index this was inserted at
                        edge_index = edge_index_list[other_vertex]

                        # The previous edge is now inactive
                        # Get pointer to prev. edge
                        pep = self.node_edge_list[nbi][edge_index][0]
                        # Insert
                        self.edge_list[pep][5] = False

                        # Update node edge list with our new edge (no appending needed)
                        self.node_edge_list[nbi][edge_index] = [ep, vp]

                        # Update the weight associated to this node.
                        lowest_weight_list[other_vertex] = eff_weight

                    else:

                        # Deactivate our edge
                        edge[5] = False

                        # Move on
                        continue

                else:
                    # Add edge to list
                    self.node_edge_list[nbi].append([ep, vp])

                    # Update the lowest_weight_list and edge_index_list
                    eff_weight = edge[0] - self.node_weight_list[node] - self.node_weight_list[other_node]
                    while self.blossom_list[other_node]:
                        other_node = self.blossom_list[other_node]
                        eff_weight -= self.node_weight_list[other_node]

                    lowest_weight_list[other_vertex] = eff_weight
                    edge_index_list[other_vertex] = num_edges
                    num_edges += 1

        # Run over edge list one last time and update the node + weight;
        # We do this after construction to avoid having to unassign
        for ep, vp in self.node_edge_list[nbi]:

            # Get edge
            edge = self.edge_list[ep]

            # Update weight while we still have the old node
            edge[0] -= self.node_weight_list[edge[4-vp]]

            # Reassign top-level node
            edge[4-vp] = nbi

    def make_instructions(self, edge_list):
        ''' Updates self.instruction_list with instructions from a list of edges

        Input:
        edge_list: a list of edges which need new instructions added

        Output:none
        '''

        for ep, vp in edge_list:

            # Get edge
            edge = self.edge_list[ep]

            # Check edge is active. It may not be, if this node
            # has two edges into another blossom and this is not
            # lowest weight.
            if not edge[5]:
                continue

            # Get nodes - we assume this_node in self.outer_node_list
            this_node = edge[4-vp]
            other_node = edge[3+vp]

            # Check that other_node is not in self.inner_node_list.
            if other_node in self.inner_node_list:
                continue

            weight = edge[0] - self.node_weight_list[other_node] - self.node_weight_list[this_node]
            if other_node in self.outer_node_list:
                outer_flag = True
                weight = weight / 2
            else:
                outer_flag = False
            # Add total growth
            weight += self.total_growth

            # Check against weight cap
            if self.weight_cap and weight > self.weight_cap+self._err:
                continue

            # Determine instruction type
            if outer_flag:  # Make blossom instruction
                inst = [weight, 2, ep, vp]
            elif self.pairing_list[self.edge_list[ep][3+vp]] and self.pairing_list[self.edge_list[ep][3+vp]][0] != 0 and self.pairing_list[self.edge_list[ep][3+vp]][0] != self.time_boundary:  # Add to tree instruction
                inst = [weight, 1, ep, vp]
            else:  # Collapse tree instruction - update weight cap
                inst = [weight, 0, ep, vp]
                self.weight_cap = weight
            self.instruction_list.put(inst)
