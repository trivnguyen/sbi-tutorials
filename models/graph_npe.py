
import torch
import torch.nn as nn
import zuko
from zuko.flows import NSF
from torch_geometric.nn import GATConv, global_mean_pool


class GraphNPE(nn.Module):
    """
    Flow-based Neural Posterior Estimation model using Neural Spline Flows
    with a Graph Attention Network (GAT) embedding network.

    This model treats stream particles as nodes in a graph and uses Graph Attention
    Networks to learn embeddings that are then fed to a normalizing flow.

    Graphs are pre-constructed during data preparation for efficiency.
    """

    def __init__(self, node_features, output_dim, hidden_size=64, embedding_size=8,
                 transforms=5, num_gat_layers=3, heads=4):
        """
        Args:
            node_features: Number of features per node (particle)
            output_dim: Dimension of output parameters
            hidden_size: Hidden dimension size for GAT layers
            embedding_size: Final embedding dimension fed to the flow
            transforms: Number of transformations in the NSF
            num_gat_layers: Number of GAT layers
            heads: Number of attention heads in GAT
        """
        super().__init__()

        self.node_features = node_features

        # Initial projection
        self.input_proj = nn.Linear(node_features, hidden_size)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            self.gat_layers.append(
                GATConv(hidden_size, hidden_size // heads, heads=heads, concat=True)
            )

        # Final projection to embedding
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size)
        )

        # Neural Spline Flow
        self.flow = NSF(
            features=output_dim,
            context=embedding_size,
            transforms=transforms,
            hidden_features=[hidden_size, hidden_size],
        )

    def embed_graph(self, batch_data):
        """
        Embed graph using GAT layers and global pooling.

        Args:
            batch_data: PyG Batch object with pre-constructed graphs

        Returns:
            embeddings: (batch_size, embedding_size)
        """
        x = batch_data.x
        edge_index = batch_data.edge_index

        # Initial projection
        x = self.input_proj(x)
        x = torch.relu(x)

        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index)
            x = torch.relu(x)

        # Global pooling to get graph-level embedding
        x_pooled = global_mean_pool(x, batch_data.batch)

        # Final projection
        embedding = self.output_proj(x_pooled)

        return embedding

    def forward(self, batch_data):
        """Return the flow conditioned on pre-constructed graph batch"""
        x_embedded = self.embed_graph(batch_data)
        return self.flow(x_embedded)

    def log_prob(self, theta, batch_data):
        """Compute log p(θ|x) for pre-constructed graph batch"""
        x_embedded = self.embed_graph(batch_data)
        return self.flow(x_embedded).log_prob(theta)

    def sample(self, batch_data, n_samples=1000):
        """Sample θ ~ p(θ|x) for pre-constructed graph batch"""
        x_embedded = self.embed_graph(batch_data)
        flow_i = self.flow(x_embedded)
        return flow_i.sample((n_samples,))
