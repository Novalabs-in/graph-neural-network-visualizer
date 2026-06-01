import torch
import torch.nn as nn
import networkx as nx

class GraphConvolutionNode(nn.Module):
    """
    GNN Layer Convolution Model
    Performs neighborhood feature aggregations for target node embeddings.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features)

    def forward(self, node_features, adjacency_matrix):
        # Aggregate neighbors: D^-1 * A * X
        degree = torch.sum(adjacency_matrix, dim=1, keepdim=True)
        deg_inv = 1.0 / (degree + 1e-9)
        normalized_adj = adjacency_matrix * deg_inv
        
        aggregated = torch.matmul(normalized_adj, node_features)
        return torch.relu(self.projection(aggregated))

if __name__ == "__main__":
    features = torch.rand((5, 8)) # 5 nodes, 8 features
    adj = torch.tensor([
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]
    ], dtype=torch.float32)
    
    gnn = GraphConvolutionNode(8, 4)
    out = gnn(features, adj)
    print("Output Node Embeddings shape:", out.shape)
