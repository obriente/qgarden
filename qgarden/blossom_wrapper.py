''' blossom_wrapper: a wrapper for the blossom algorithm written
for this package.

(c) 2017 Thomas O'Brien
Distributed under the GNU GPLv3. See LICENSE.txt or
https://www.gnu.org/licenses/gpl.txt

The blossom algorithm used in the qec package is written to integrate
with the gardener that sits over the top of it. The following provides
a wrapper for people who want to use this implementation of blossom
for other reasons. It takes as input a standard weight matrix, along
with a cutoff (in case you don't want to feed in too-large weights).

Note that my implementation of blossom is quite slow. The only advantage
that this code has over the many better packages that can be found on the
internet is really the implementation of the boundary. As such, this code
will insist upon row 0 corresponding to a boundary vertex. Note that adding
negative weights in the first row will thus cause the algorithm to fail.
'''

from . import blossom


def insert_wm(wm, cutoff=None, null_flag=False):
    '''
    Converts a weight matrix into a string of vertices to feed into a
    constantly updating version of blossom, inserts it, and returns result.

    Assumes that the first index corresponds to the boundary,

    Input:s
    wm: a matrix in the form of a list of lists
    cutoff: maximum weight to feed the algorithm. If None, feeds all weights.
    null_flag: whether wm[i,j]=0 corresponds to no edge (False) or a weight-
        zero edge (True).
    '''

    b = blossom.Bloss(tbw=1e9)

    # Run over weight matrix
    for index in range(1, len(wm)):

        # Pull list of weights to insert
        weight_list = wm[index][:index]

        # initialize the set that we will send to blossom
        # we always include the edge to the boundary
        weight_set = [[0, weight_list[0]]]

        # Run over list of weights, turn them into the correct format
        # and insert them into weight_set, if they are sufficiently small
        # weight. We also assume zero weights correspond to 'no edge'
        for index2, weight in zip(range(1, index), weight_list[1:]):

            if null_flag:
                if weight <= 0 or (cutoff and weight > cutoff):
                    continue
            else:
                if weight < 0 or (cutoff and weight > cutoff):
                    continue

            weight_set.append([index2, weight])

        # Insert into blossom
        b.add_vertex(weight_set)

    pairing = b.finish()

    return pairing
