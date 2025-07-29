import torch
import numpy as np

class DynamicGraphConstructor:
    def __init__(self, window_size, stride=1):
        self.window_size = window_size
        self.stride = stride

    def compute_adjacency(self, window):
        """Compute adjacency matrix using Pearson correlation."""
        corr = np.corrcoef(window.T)  # N x N correlation matrix
        corr = np.nan_to_num(corr, 0)  # Handle NaNs
        adj = torch.tensor(corr, dtype=torch.float)
        adj = torch.where(adj > 0.5, adj, torch.zeros_like(adj))  # Threshold for sparsity
        return adj

    def to_dynamic_graph(self, time_series):
        """Convert time-series (T x N) to dynamic graphs."""
        T, N = time_series.shape
        graphs = []
        for t in range(0, T - self.window_size + 1, self.stride):
            window = time_series[t:t + self.window_size, :]
            adj = self.compute_adjacency(window)  # A_t
            features = torch.tensor(window.T, dtype=torch.float)  # N x W
            graphs.append((adj, features))
        return graphs

# Example usage
if __name__ == "__main__":
    data = np.random.rand(100, 5)  # T=100, N=5
    constructor = DynamicGraphConstructor(window_size=10)
    graphs = constructor.to_dynamic_graph(data)
    print(f"Generated {len(graphs)} graphs")