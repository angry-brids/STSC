import torch
import random

class GraphAugmenter:
    def __init__(self, edge_dropout_prob=0.2, noise_scale=0.1):
        self.edge_dropout_prob = edge_dropout_prob
        self.noise_scale = noise_scale

    def edge_dropout(self, adj):
        """Randomly drop edges from adjacency matrix."""
        mask = torch.rand_like(adj) > self.edge_dropout_prob
        return adj * mask

    def feature_noise(self, feats):
        """Add Gaussian noise to features."""
        noise = torch.randn_like(feats) * self.noise_scale
        return feats + noise

    def augment(self, graphs):
        """Generate an augmented view of the graph sequence."""
        aug_graphs = []
        for adj, feats in graphs:
            aug_adj = self.edge_dropout(adj)
            aug_feats = self.feature_noise(feats)
            aug_graphs.append((aug_adj, aug_feats))
        return aug_graphs

# Example usage
if __name__ == "__main__":
    graphs = [(torch.rand(5, 5), torch.rand(5, 10))] * 3
    augmenter = GraphAugmenter()
    aug_graphs = augmenter.augment(graphs)
    print(len(aug_graphs))