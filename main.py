import argparse
import numpy as np
import networkx as nx
import pandas as pd
import node2vec
import matplotlib.pyplot as plt

from word2vec import learn_embeddings
from tools import visualize_emb


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--ratings-file', nargs='?', default='data/test_data_long',
                        help='Input data path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--num_dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    
    parser.add_argument('--num-negative-samples', type=int, default=50,
                        help='Number of nodes (not contained in a given walk) used for negative sampling. Default is 50.')

    parser.add_argument('--num_epochs', default=1, type=int,
                        help='Number of epochs in SGD')
    
    parser.add_argument('--learning-rate', default=0.01, type=int,
                        help='Learning rate in SGD')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    return parser.parse_args()


def _test_basic_embedding(emb, G):
    # TODO do something more formal
    """Compute statistics for embedding of a simple undirected graph without weights"""
    n = len(G.nodes)
    dist_conn = [np.linalg.norm(emb[e[0]] - emb[e[1]]) for e in G.edges]
    
    non_edges = [[x,y] for x in range(n) for y in range(x+1,n) if [x,y] not in G.edges]
    dist_disconn = [np.linalg.norm(emb[e[0]] - emb[e[1]]) for e in non_edges]
    
    # Average distance between each pair of adjacent and each pair of non-adjacent nodes
    mean_conn = np.array(dist_conn).mean()
    mean_disconn = np.array(dist_disconn).mean()

    
    all_dist = np.array([np.linalg.norm(emb[x] - emb[y]) for x in range(n) for y in range(x+1,n)])
    adjacency = np.array([([x,y] in G.edges) for x in range(n) for y in range(x+1,n)])
    # Correlation between distance and adjacency. Negative correlation indicates that the embeddings of adjacent nodes are close. 
    corr = np.corrcoef(all_dist, adjacency)


def main(args):
    # Read Facebook graph
    G = nx.read_edgelist('data/facebook_combined.txt', nodetype=int, data=[('weight', int)])
    # The node2vec algorithm can also work with edge weights. In this case, everything is set to 1
    nx.set_edge_attributes(G, values=1, name='weight')
    # TODO separate edges into training and test sets

    walker = node2vec.Node2Vec(G, args.p, args.q)
    walker.preprocess_transition_probs()
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    n = len(G.nodes)
    emb = learn_embeddings(walks, n, args)
    _test_basic_embedding(emb, G)
    visualize_emb(emb, G)



if __name__ == "__main__":
    args = parse_args()
    main(args)