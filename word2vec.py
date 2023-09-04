import torch 
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Node2VecLoss(nn.Module):
    def __init__(self):
        super(Node2VecLoss, self).__init__()

    def forward_old(self, embedding, source_node, context_nodes, neg_samples):
        source_embedding = embedding(source_node)
        context_embedding = embedding(context_nodes)
        neg_samples_embedding = embedding(neg_samples)
        
        positives = context_embedding @ source_embedding.t()
        positives = torch.sum(positives)

        negatives = neg_samples_embedding @ source_embedding.t()
        negatives = torch.exp(negatives)
        negatives = torch.log(torch.sum(negatives))
        # there should at least be normalization by a factor of ((n-len(context_nodes)/num_negative_samples))

        loss = positives - negatives
        return loss
    
    def forward(self, embedding, source_node, context_nodes, neg_samples):
        # Get embeddings
        source_embedding = embedding(source_node).squeeze()
        context_embedding = embedding(context_nodes)
        neg_samples_embedding = embedding(neg_samples)

        # Positive pair similarity
        positives = torch.sigmoid(torch.sum(context_embedding @ source_embedding))
        positives = torch.clamp(positives, min=1e-7, max=1-1e-7)
        
        # Negative pair similarity
        negatives = torch.sigmoid(-torch.mm(neg_samples_embedding, source_embedding.unsqueeze(1))).sum()
        negatives = torch.clamp(negatives, min=1e-7, max=1-1e-7)

        # Loss
        loss = -torch.log(positives) - negatives

        return loss

    

def learn_embeddings(random_walks, G, embedding_size=10, num_negative_samples=5, lr=0.01, epochs=100):
    n_nodes = len(G.nodes)
    embedding = nn.Embedding(n_nodes, embedding_size)
    nn.init.xavier_uniform_(embedding.weight)
    criterion = Node2VecLoss()
    optimizer = optim.Adam(embedding.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for walk in random_walks:
            if len(walk) == 1: 
                continue

            source_node = torch.tensor(walk[0])  # TODO use intermediate nodes as start as well
            context = torch.tensor(walk[1:])

            # Zero the gradients
            optimizer.zero_grad()

            # Negative sampling
            size = min(num_negative_samples, n_nodes-len(walk))
            assert size > 0
            neg_samples = np.random.choice([n for n in range(n_nodes) if n not in walk], size=size, replace=False)
            neg_samples = torch.tensor(neg_samples)

            # Compute the loss
            loss = criterion(embedding, source_node, context, neg_samples)
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), max_norm=1.0)

            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    res = embedding(torch.arange(n_nodes)).detach().numpy()
    return res

"""
# Example usage
n = 4
G = nx.path_graph(n)
G = G.to_undirected()
random_walks = [[0,1], [1,2], [1,2,3]]
emb = learn_embedding(random_walks, G, embedding_size=2, lr=0.1, epochs=10)

all_embeddings = emb(torch.arange(n)).detach().numpy()
positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)
"""

# all_embeddings = emb(torch.arange(n)).detach().numpy()
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(all_embeddings)
# positions = {i: (pca_result[i, 0], pca_result[i, 1]) for i in range(n)}

# nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)
# plt.show()



# n_emb = 3
# n_node = 4

# embeddings = torch.arange(1, n_node*n_emb+1).view((n_node, n_emb))

# node = 0
# neighbors = torch.tensor([1,2])
# neg_samples = torch.tensor([2,3])

# positives = embeddings[neighbors,:] @ embeddings[node,:]
# positives = torch.sum(positives)

# negatives = embeddings[neg_samples,:] @ embeddings[node,:]
# negatives = torch.exp(negatives)
# negatives = torch.log(torch.sum(negatives))

# loss = positives - negatives




# embeddings = nn.Embedding(n_node, n_emb)

# node = 0
# neighbors = torch.tensor([1,2])
# neg_samples = torch.tensor([2,3])

# node_emb = embeddings(torch.tensor([node]))
# context_emb = embeddings(neighbors)
# neg_emb = embeddings(neg_samples)

# positives = context_emb @ node_emb.t()
# positives = torch.sum(positives)

# negatives = neg_emb @ node_emb.t()
# negatives = torch.exp(negatives)
# negatives = torch.log(torch.sum(negatives))

# loss = positives - negatives