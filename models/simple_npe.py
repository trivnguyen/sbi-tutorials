
import torch
import torch.nn as nn
import zuko
from zuko.flows import NSF

class SimpleNPE(nn.Module):
    """ Flow-based Neural Posterior Estimation model using Neural Spline Flows with an MLP embedding network. """

    def __init__(self, input_dim, output_dim, hidden_size=64, embedding_size=8, transforms=5):
        super().__init__()

        self.embedding_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        self.flow = NSF(
            features=output_dim,
            context=embedding_size,
            transforms=transforms,
            hidden_features=[hidden_size, hidden_size],
        )

    def forward(self, x):
        """Return the flow conditioned on x"""
        x_embedded = self.embedding_network(x)
        return self.flow(x_embedded)

    def log_prob(self, theta, x):
        """Compute log p(θ|x)"""
        x_embedded = self.embedding_network(x)
        return self.flow(x_embedded).log_prob(theta)

    def sample(self, x, n_samples=1000):
        """Sample θ ~ p(θ|x)"""
        x_embedded = self.embedding_network(x)
        flow_i = self.flow(x_embedded)
        return flow_i.sample((n_samples,))
