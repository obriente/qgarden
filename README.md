# QGarden - release edition

A fully-python blossom decoder for use in decoding surface and repetition quantum error correcting codes.

Authors: Thomas O'Brien (obrien@lorentz.leidenuniv.nl), Boris Varbanov, Stephen Spitz

Thanks to: Brian Tarasinski, Adriaan Rol, Niels Bultink, Xiang Fu, Ben Criger, Paul Baireuther and Leo DiCarlo for comments and assistance.

References: The article to cite when using this code is
	O'Brien, T.E. and Tarasinski, B. and DiCarlo, L. npj Quant. Inf. 3, 39 (2017)
## Overview
MWPM decoding for surface and repetition codes consists of three reasonably-well separated tasks:
1. Generating weight data

2. Translating measurement output to and from the minimum-weight perfect matching (MWPM) problem

3. Solving the MWPM problem

This library provides three separate programs to solve these three tasks. The second two libraries are intended to be plug-and-play for any distance and error model (with the second providing a complete interface between the user and the third). The first problem is model-dependent; we provide a program that may require some re-writing on the behalf of the user to suit their needs.

## Generating weight data (Task 1)
The weight generation currently in this library is still a bit incomplete. In general, one may take an error model, assign all error probabilities to edges between vertices in the 'ancilla graph' (using the notation of arXiv:1703.04136) or 'primal graph' (using the notation of arXiv:1202.6111) and use the inversion formula (Eqn. C2 of arXiv:1703.04136) to generate the required weight matrix. However, we do not yet have a good way of automating this assignment. So, dear user, you are on your own for a bit here.

We provide a file 'weight_gen_simple' which provides the weight matrices used for the model of arXiv:1705.07855 . Hopefully it is well-commented enough to be understandible.

One final thing that needs to be provided is a correction matrix. This tells blossom whether any given error chain commutes or does not commute with your favourite X (or Z) logical operator, which in turn determines the parity correction from any given matching. This is made for you in weight_gen_simple.get_correction_matrix() , and is only dependent on geometry and the choice of logical operator, as opposed to the underlying error model.

## Taking care of your very own blossom (Task 2)
Proper use of the blossom algorithm requires feeding it a good selection of weights and vertices, running it when necessary, and harvesting the final pairing to turn into a beautiful error correction. In order to perform all of these tasks, we have provided you with a gardener file (because bad puns, coffee and alcohol are the backbone of physics).

### How to use the gardener
For the convenience of the user, we have provided a file 'decoder_template' that demonstrates how to use the gardener. But it's a very simple process:

1. Initialise the gardener (g=gardener.Gardener()), with appropriate input.
2. Insert rounds of stabilizer measurement data (gardener.update()).
3. Insert a round of final stablizer measurements (extracted from any final readout) and receive a parity measurement (gardener.result()). This may be called with a continue_flag=True if you wish to store the gardener state before this final measurement is calculated, and then continue to insert more stabilizer data.
4. If you want to reuse your gardener for a new QEC experiment, use gardener.reset().

## The MWPM problem (Task 3)
The minimum-weight perfect matching problem is famously solved by Edmonds' blossom algorithm in polynomial time. We provide a homebrew python implementation of this in the blossom file. If you want to do your own gardening for whatever reason, please feel free to direct-access this instead!

### How to use blossom.py
For the convenience of the user, we have provided a small wrapper file (aptly named blossom_wrapper) which takes as input a weighted adjacency matrix and feeds it into the blossom class. This can hopefully serve as a good example of how to do this.

blossom.py contains a Blossom class, which has the following important functions:

init(time_boundary_weight): Initialise a graph with only a boundary vertex. time_boundary_weight dictates the amount by which a vertex may grow in weight per time-step (preventing the overtightening of edges in a QEC experiment).

add_vertex(weight_list): Add a vertex to the graph, along with edges connecting it to vertices already in the graph. Weight_list should contain pairs of indices (index, weight), where index is the index of the other vertex in order of insertion. **Note** the boundary is pre-inserted and always has index 0.

run(): Runs the blossom algorithm until either all vertices are matched (excepting the boundary), or a vertex weight attempts to grow beyond its 'time_weight'.

add_t_weight(): Increases the 'time_weight' on each vertex by the pre-determined time_boundary_weight.

finish(weight_lists, continue_flag): Inserts a potential final round of vertices and edges into the graph (as determined by weight_lists), and completes the blossom algorithm, ignoring any time weights on any vertices. Returns the resultant matching to the user. If continue_flag=True, stores the state of the class before this, and restores it afterwards. This allows for simulation experiments where multiple hypothetical readouts may be performed, rather than a physical experiment where this collapses the system state.

### Differences to other blossom implementations
This deviates from standard implementations of the blossom algorithm in the following ways:

1. The inclusion of a boundary vertex that may be paired to any number of other vertices (including 0). This is a standard adaptation for QEC (see for instance arXiv:1202.6111).

2. The ability to add vertices to a graph mid-algorithm. As long as no added edge is overtight, the blossom algorithm has no native problem with this (this is hardly a new idea either).

3. Speed-wise, this is missing many of the features of say, Blossom V, and consequently is reasonably slow.

### Known issues
Our blossom algorithm has the following known bugs(/features):

1. We assume every other vertex has an edge to the boundary, which implies that in a matching *all* vertices will be paired. This is different from the standard MWPM problem, and currently the code will throw errors if this is not the case.

2. Currently feeding in vertices with overtight edges will cause the algorithm to fail. In the future it would be good to make it so that instead we just shrink the appropriate vertices and unmatch the appropriate edges. 

### License
This work is distributed under the GNU GPLv3. See LICENSE.txt. (c) 2017 Thomas O'Brien
