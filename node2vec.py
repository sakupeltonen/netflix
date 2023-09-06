import numpy as np
import networkx as nx
import random


"""
TODO
- basic set up on some simple graph, e.g. karate club
- adapt to bipartite graphs
- adapt to graphs with signed weights


Adapting to bipartite signed graphs: 
=========================================
Random walks on the projection graphs, where users (items) are adjacent if they have a common item (user).
The weight of an edge is the product of the weights. Takes care of the sign automatically. 
Do we need actually 4 walks,  (sign x partition)
Only do two walks, one for each partition. The virtual edges consisting of edges with the same sign.


The number of edges in the projection graph is potentially quite large

"""

class Node2Vec():
    """
    Node2Vec implementation from https://github.com/aditya-grover/node2vec/tree/master

    Modified for bipartite graphs with signed weights
    """
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.

        Modification: store two sides of the walk, both containing user and item nodes possibly. 
        The walk starts on one side and changes to the other side whenever an edge with a negative sign is traversed. 
        The sign is given by the rating of user for movie, after normalizing the users ratings
        The edge weights are proportional to absolute value of normalized ratings. 

        The motivation for this is to avoid constructing a projected graph from the bipartite representation, which would be very expensive. 
        Also, this allows walks to contain both users and movies. 
        
        Intuitively, the enemy of my enemy is my friend: switching sides twice (in a row) should give similar nodes again
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        # sided_walk = ([start_node], [])
        # side = 0
        walk = [start_node]

        k = 1  # counter for the length of the walk

        #_prev = (None, None)  # previous node for both sides of the bipartition
        #_cur = (start_node, None)  # current node for both sides of the bipartition
        #p = 0  # can start in either users or movies
        prev = None
        cur = start_node

        while k < walk_length:
            #cur = _cur[p]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if k == 1:
                    # alias_nodes[cur][0], alias_nodes[cur][1] gives J, q for current node
                    next = cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])] 
                else:
                    # prev = _prev[p]  # doesn't work: we don't have alias set up for virtual edges
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                
                # Switch sides if traversing a negative edge
                # if G[cur][next]['signed_weight'] < 0: 
                #     side = 1 - side
                # sided_walk[side].append(next)
                walk.append(next)

                k += 1
                prev = cur
                cur = next

            else:
                break

        # return sided_walk
        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # sided_walk = self.node2vec_walk(walk_length=walk_length, start_node=node)    
                # walks.append(sided_walk[0])
                # if len(sided_walk[1]) > 0:
                #     walks.append(sided_walk[1])

                walk = self.node2vec_walk(walk_length=walk_length, start_node=node)
                walks.append(walk)
        return walks

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        # // these are only needed for the first step of each walk
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int32)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def alias_draw(J, q):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]
