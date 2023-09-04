import argparse
import numpy as np
import networkx as nx
import pandas as pd
import node2vec
import random
import torch
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

from word2vec import learn_embeddings
from preprocess import loadData, filterByRatingCount

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")

    # parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
    #                     help='Input graph path')

    parser.add_argument('--output', nargs='?', default='emb/karate.emb',
                        help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--max-projected-edges', type=int, default=50,
                        help='Maximum number of edges constructed in the projection graph for a single edge in the bipartite user-item graph.')

    # parser.add_argument('--directed', dest='directed', action='store_true',
    #                     help='Graph is (un)directed. Default is undirected.')
    # parser.add_argument('--undirected', dest='undirected', action='store_false')
    # parser.set_defaults(directed=False)

    return parser.parse_args()


def get_karate_club():
    G = nx.karate_club_graph()
    G = G.to_undirected()
    for edge in G.edges():
        # // this is the correct way to access edge attributes 
        G[edge[0]][edge[1]]['weight'] = 1
        G[edge[0]][edge[1]]['signed_weight'] = 1
    return G


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    file = 'data/test_data_long'
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
    #G.add_nodes_from(df['customer'],bipartite=0)
    #G.add_nodes_from(df['movie'],bipartite=1)
    global_average_rating = df['rating'].mean()
    for _, row in df.iterrows():
        customer = row['customer']
        average_rating = df_customers.loc[customer, 'average']
        weight_normalized = row['rating'] - (average_rating + global_average_rating)/2
        # TODO handle the case when this is zero. 
        # wouldn't want to not add the edge, because the resulting graph will be quite sparse in most cases
        # // fixed majority of cases by averaging with the global average
        if weight_normalized != 0:
            G.add_edge(customer, row['movie'], weight=abs(weight_normalized), signed_weight=weight_normalized)
    

    # G_customer = nx.bipartite.projected_graph(G, list(df['customer']))
    # the weight of a single connection should be the product
    # how to handle multiple connections with different products? could just average them

    # constructing the graph explicitly seems kind of dumb
    # should look at the bipartite network embedding paper
    # could only consider a subgraph of the projection graph, by sampling a random set of users for each user and movie. could create some imbalances though, for example might be relevant to have info about how many customers have rated a specific movie

    """
    # TODO seems to get stuck somewhere
    G_customer = nx.MultiGraph()
    G_customer.add_nodes_from(df['customer'])
    for cust_a in G_customer:
        # TODO avoid adding duplicate edges
        rated_movies = df[df['customer'] == cust_a]
        for _, row in rated_movies.iterrows():
            movie = row['movie']
            weight_a = G[cust_a][movie]['weight']
            relations = df[df['movie'] == movie]
            relations = relations[relations['customer'] != cust_a]  # remove the row corresponding to original user
            other_customers = list(relations['customer'])

            sampled_customers = random.sample(other_customers, min(len(other_customers), args.max_projected_edges))
            
            relations = relations[relations['customer'].isin(sampled_customers)]
            for _, row in relations.iterrows():
                cust_b = row['customer']
                weight_b = G[cust_b][movie]['weight']
                total_weight = weight_a * weight_b
                if total_weight > 0:
                    # i.e. weights have the same sign
                    G_customer.add_edge(cust_a, cust_b, weight=total_weight)
                    # there may be multi-edges. the random walk handles this quite naturally. 
                    # TODO what about the case when weights have different signs
                G = G.to_undirected()
    
    # TODO same for movies
    # - could easily write same code in a more general way, but might make it quite unreadable
    """
    # ANOTHER IDEA: random walk with weights proportional to absolute value of normalized rating
    # - for each random walk, maintain two lists as a result
    # walk starts to add nodes to list A, change the active list whenever traversing a negative edge
    # intuition: traversing negative edge twice means that both users dislike the same movie

    return G

def learn_embeddings_old(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
    model.save(args.output)
    
    return model

def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    # nx_G = read_graph()
    # nx_G = get_karate_club()
    n = 20
    nx_G = nx.cycle_graph(30)
    for edge in nx_G.edges():
        # // this is the correct way to access edge attributes 
        nx_G[edge[0]][edge[1]]['weight'] = 1
        nx_G[edge[0]][edge[1]]['signed_weight'] = 1

    args.walk_length = n//3
    
    walker = node2vec.Node2Vec(nx_G, False, args.p, args.q)  # TEMP directed=False hardcoded
    walker.preprocess_transition_probs()
    walks = walker.simulate_walks(args.num_walks, args.walk_length)

    emb = learn_embeddings(walks, nx_G, embedding_size=2, num_negative_samples=round(n/4), epochs=50)
    visualize_emb(emb, nx_G)
    draw(emb, nx_G)
    # TODO current implementation doesn't use the fact that the graph is bipartite at all
    #  - could just keep track of which partition the walk is on, and use connectivity of the last node on that side.
    #  - the sided walk does give enemy of my enemy is my friend -type connections, but it doesn't increase the distance between two enemies in any way
    # TODO recommendations. can do this even if the model isn't perfect
    

def draw(embeddings, G):
    n = len(G.nodes)
    
    # positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
    positions = {i: (embeddings[i, 0], embeddings[i, 1]) for i in range(n)}
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)

    plt.show()

def visualize_emb(embeddings, G):
    n = len(G.nodes)
    
    # positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    positions = {i: (pca_result[i, 0], pca_result[i, 1]) for i in range(n)}
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)

    plt.show()



# if __name__ == "__main__":
#     args = parse_args()
#     main(args)

args = parse_args()
args.walk_length = 10
args.num_walks = 5
args.workers = 1
args.q = 3
main(args)