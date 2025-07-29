import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGPooling
from torch.nn import GRU, TransformerEncoder, TransformerEncoderLayer

class STHAFEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_heads=4, gat_layers=2, transformer_layers=2, pool_clusters=8):
        super(STHAFEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        # Multi-Head GAT
        self.gat_layers = nn.ModuleList()
        for _ in range(gat_layers):
            in_channels = in_dim if _ == 0 else hidden_dim * num_heads
            self.gat_layers.append(GATConv(in_channels, hidden_dim, heads=num_heads, concat=True))

        # Transformer-Augmented GRU
        self.gru = GRU(hidden_dim * num_heads, hidden_dim, batch_first=True)
        transformer_layer = TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2)
        self.transformer = TransformerEncoder(transformer_layer, num_layers=transformer_layers)

        # Hierarchical Pooling
        self.pool = SAGPooling(hidden_dim, ratio=pool_clusters / in_dim)
        self.temporal_attention = nn.MultiheadAttention(hidden_dim, num_heads=1)

    def forward(self, graphs):
        """Process dynamic graph sequence."""
        spatial_embeds = []
        for adj, feats in graphs:  # G_t = (A_t, F_t)
            x = feats
            edge_index = adj.nonzero(as_tuple=False).t()  # Convert adj to edge_index
            for gat in self.gat_layers:
                x = gat(x, edge_index).relu()
            spatial_embeds.append(x)  # N x (hidden_dim * num_heads)

        # Temporal encoding
        spatial_seq = torch.stack(spatial_embeds, dim=0)  # T' x N x H
        gru_out, _ = self.gru(spatial_seq)  # T' x N x hidden_dim
        transformer_out = self.transformer(gru_out.transpose(0, 1))  # N x T' x hidden_dim

        # Hierarchical pooling
        pooled_embeds = []
        for t in range(transformer_out.size(1)):
            x = transformer_out[:, t, :]
            edge_index = graphs[t][0].nonzero(as_tuple=False).t()
            x, _, _, _, _, _ = self.pool(x, edge_index)  # C x hidden_dim
            pooled_embeds.append(x)
        pooled_seq = torch.stack(pooled_embeds, dim=0)  # T' x C x hidden_dim

        # Temporal attention
        attn_out, _ = self.temporal_attention(pooled_seq, pooled_seq, pooled_seq)
        z = attn_out.mean(dim=0)  # C x hidden_dim
        return z

# Example usage
if __name__ == "__main__":
    model = STHAFEncoder(in_dim=10, hidden_dim=16)
    adj = torch.rand(5, 5)
    feats = torch.rand(5, 10)
    graphs = [(adj, feats)] * 8  # Dummy sequence
    z = model(graphs)
    print(z.shape)