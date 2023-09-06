import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Node2VecLoss(nn.Module):
    def __init__(self):
        super(Node2VecLoss, self).__init__()
    
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

    

def learn_embeddings(random_walks, n, args):
    """
       Compute embeddings given a list of random walks
    
       Parameters:
       random_walks ([[int]]): List of random walks
       n (int): Number of nodes in the original graph. Node identifiers are assumed to be 0,...,n-1
       args: See embedding.py
    
       Returns:
       res: nn.Embedding
    """
    embedding = nn.Embedding(n, args.num_dimensions)
    nn.init.xavier_uniform_(embedding.weight)
    criterion = Node2VecLoss()
    optimizer = optim.Adam(embedding.parameters(), lr=args.learning_rate)

    for epoch in range(args.num_epochs):
        total_loss = 0
        for walk in random_walks:
            if len(walk) == 1: 
                continue

            source_node = torch.tensor(walk[0])  # TODO use intermediate nodes as start as well
            context = torch.tensor(walk[1:])

            # Zero the gradients
            optimizer.zero_grad()

            # Negative sampling
            size = min(args.num_negative_samples, n-len(walk))
            assert size > 0
            neg_samples = np.random.choice([n for n in range(n) if n not in walk], size=size, replace=False)
            neg_samples = torch.tensor(neg_samples)

            # Compute the loss
            loss = criterion(embedding, source_node, context, neg_samples)
            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), max_norm=1.0)

            optimizer.step()
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Loss: {total_loss:.4f}")

    res = embedding(torch.arange(n)).detach().numpy()
    return res

"""
# Example usage
n = 4
G = nx.path_graph(n)
G = G.to_undirected()
random_walks = [[0,1], [1,2], [1,2,3]]
emb = learn_embedding(random_walks, G, embedding_size=2, lr=0.1, num_epochs=10)

all_embeddings = emb(torch.arange(n)).detach().numpy()
positions = {i: tuple(all_embeddings[i,:]) for i in range(n)}
nx.draw(G, pos=positions, with_labels=True, node_color='lightblue', node_size=400)
"""