from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt


def visualize_emb(embeddings, G):
    """Visualize an embedding of the nodes by computing a PCA with 2 components"""
    n = len(G.nodes)
    
    # positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeddings)
    positions = {i: (pca_result[i, 0], pca_result[i, 1]) for i in range(n)}
    nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)

    plt.show()