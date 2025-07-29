import torch
import torch.nn.functional as F

class SpatioTemporalContrastiveLoss:
    def __init__(self, temperature=0.2):
        self.temperature = temperature

    def compute_weight(self, graphs):
        """Compute spatio-temporal weighting based on edge density."""
        densities = []
        for adj, _ in graphs:
            density = adj.sum() / (adj.size(0) * adj.size(1))
            densities.append(density)
        weights = torch.tensor(densities, dtype=torch.float)
        return weights / weights.max()

    def forward(self, z1, z2, graphs1, graphs2):
        """Compute NT-Xent loss with weighting."""
        B = z1.size(0)  # Batch size
        z = torch.cat([z1, z2], dim=0)  # 2B x K
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature  # 2B x 2B

        # Positive pairs: (z1[i], z2[i])
        pos_mask = torch.eye(B, dtype=torch.bool, device=z.device)
        pos_mask = torch.cat([pos_mask, pos_mask], dim=1)
        pos_mask = torch.cat([pos_mask, pos_mask], dim=0)

        # Negative pairs: all others
        neg_mask = ~pos_mask

        pos_sim = sim_matrix[pos_mask].view(2 * B, 1)
        neg_sim = sim_matrix[neg_mask].view(2 * B, -1)

        # Loss
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
        base_loss = F.cross_entropy(logits, labels)

        # Weighting
        w1 = self.compute_weight(graphs1)
        w2 = self.compute_weight(graphs2)
        weights = torch.cat([w1, w2])
        weighted_loss = (weights * base_loss).mean()
        return weighted_loss

# Example usage
if __name__ == "__main__":
    z1 = torch.rand(4, 128)
    z2 = torch.rand(4, 128)
    graphs = [(torch.rand(5, 5), torch.rand(5, 10))] * 4
    loss_fn = SpatioTemporalContrastiveLoss()
    loss = loss_fn(z1, z2, graphs, graphs)
    print(loss.item())