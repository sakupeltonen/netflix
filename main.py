import argparse
import numpy as np
import networkx as nx
import pandas as pd
import node2vec
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from word2vec import learn_embeddings
from preprocess import loadData, filterByRatingCount

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


def main(args):
    n = 100
    G = nx.cycle_graph(n)
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1

    walker = node2vec.Node2Vec(G, args.p, args.q)
    walker.preprocess_transition_probs()
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    emb = learn_embeddings(walks, n, args)
    _test_basic_embedding(emb, G)
    visualize_emb(emb, G)
    


def visualize_emb(embeddings, G):
    """Visualize an embedding of the nodes by computing a PCA with 2 components"""
    n = len(G.nodes)
    
    # positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    positions = {i: (pca_result[i, 0], pca_result[i, 1]) for i in range(n)}
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)

    plt.show()


def _test_basic_embedding(emb, G):
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

# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

args = parse_args()
args.walk_length = 10
args.num_walks = 30
args.num_dimensions = 2
args.p = 0.5
args.q = 2
main(args)



def _read_netflix_graph(file):
    """Create a graph from netflix user ratings. Needs to be integrated to the node2vec functions"""
    df = loadData(file)
    df = filterByRatingCount(df, 30, 200)
    df['customer'] = 'c' + df['customer']
    df['movie'] = 'm' + df['movie'].astype(str)
    

    # Create a df of customers
    means = df.groupby('customer')['rating'].mean()
    counts = df.groupby('customer')['rating'].count()
    df_customers = pd.concat([means, counts], axis=1)
    df_customers.columns = ['average','count']

    # Construct bipartite graph of normalized ratings
    G = nx.Graph()
    global_average_rating = df['rating'].mean()
    for _, row in df.iterrows():
        customer = row['customer']
        average_rating = df_customers.loc[customer, 'average']
        # weight_normalized = row['rating'] - average_rating
        weight_normalized = row['rating'] - (average_rating + global_average_rating)/2
        if weight_normalized != 0:
            G.add_edge(customer, row['movie'], weight=abs(weight_normalized), signed_weight=weight_normalized)

    return G