import argparse
import numpy as np
import networkx as nx
import pandas as pd
import node2vec
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
from word2vec import save_embedding

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

from word2vec import learn_embeddings
from tools import visualize_emb


def parse_args():
    parser = argparse.ArgumentParser(description="Run node2vec.")
    
    parser.add_argument('--data-relative-path', default='data/facebook_combined.txt',
                        help='Input data path')

    parser.add_argument('--save-embedding', action='store_true')

    parser.add_argument('--num-dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    
    parser.add_argument('--num-negative-samples', type=int, default=50,
                        help='Number of nodes (not contained in a given walk) used for negative sampling. Default is 50.')

    parser.add_argument('--num-epochs', default=1, type=int,
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
    G = nx.read_edgelist(args.data_relative_path, nodetype=int)

    train_edges, test_edges = train_test_split(list(G.edges), test_size=0.1, random_state=42)

    train_G = nx.Graph()
    train_G.add_edges_from(train_edges)

    test_G = nx.Graph()
    test_G.add_edges_from(test_edges)

    walker = node2vec.Node2Vec(train_G, args.p, args.q)
    walker.preprocess_transition_probs()
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    n = len(G.nodes)
    # not all nodes are included in train_G. the embedding of those nodes is arbitrary
    emb = learn_embeddings(walks, n, args)

    if args.save_embedding:
        save_embedding(emb, training_args=vars(args))
    
    # link prediction
    non_edges = list(nx.non_edges(G))
    negative_samples = random.sample(non_edges, 10*len(test_edges))  # TODO should train on many more negative samples than positive samples
    negative_train_samples = negative_samples[:len(negative_samples)//2]
    negative_test_samples = negative_samples[len(negative_samples)//2:]

    train_data = train_edges + negative_train_samples
    test_data = test_edges + negative_test_samples

    for u,v in test_data:
        if u not in train_G.nodes or v not in train_G.nodes:
            test_data.remove((u,v))

    X_train = np.zeros((len(train_data), 2*args.num_dimensions))
    y_train = np.zeros(len(train_data))

    X_test = np.zeros((len(test_data), 2*args.num_dimensions))
    y_test = np.zeros(len(test_data))

    for i, edge in enumerate(train_data):
        u, v = edge
        # Concatenate, average, or perform any operation to combine embeddings. # TODO maybe just dot product
        feature_vector = np.concatenate((emb[u], emb[v]))

        X_train[i] = feature_vector
        y_train[i] = 1 if edge in train_edges else 0

    for i, edge in enumerate(test_data):
        u, v = edge
        feature_vector = np.concatenate((emb[u], emb[v]))
        
        X_test[i] = feature_vector
        y_test[i] = 1 if edge in test_edges else 0

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]  # probability estimates of the positive class

    # Evaluation
    print("Classification Report:\n", classification_report(y_test, y_pred))  # TODO what are all the results
    print("AUC-ROC: ", roc_auc_score(y_test, y_prob))  # TODO what does this tell

    # TODO general prediction function for a given node pair





if __name__ == "__main__":
    args = parse_args()
    main(args)