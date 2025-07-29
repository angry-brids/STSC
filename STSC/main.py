import torch
import torch.optim as optim
from graph_construction import DynamicGraphConstructor
from sthaf_encoder import STHAFEncoder
from contrastive_loss import SpatioTemporalContrastiveLoss
from augmentations import GraphAugmenter
import numpy as np

# Hyperparameters
WINDOW_SIZE = 10
HIDDEN_DIM = 16
BATCH_SIZE = 4
EPOCHS = 10

# Dummy data (replace with real dataset)
data = np.random.rand(100, 5)  # T x N
constructor = DynamicGraphConstructor(window_size=WINDOW_SIZE)
graphs = constructor.to_dynamic_graph(data)

# Model and components
model = STHAFEncoder(in_dim=WINDOW_SIZE, hidden_dim=HIDDEN_DIM)
augmenter = GraphAugmenter()
loss_fn = SpatioTemporalContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for i in range(0, len(graphs) - BATCH_SIZE + 1, BATCH_SIZE):
        batch_graphs = graphs[i:i + BATCH_SIZE]

        # Augmentations
        aug1 = augmenter.augment(batch_graphs)
        aug2 = augmenter.augment(batch_graphs)

        # Forward pass
        z1 = model(aug1)
        z2 = model(aug2)

        # Loss
        loss = loss_fn(z1, z2, aug1, aug2)
        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss:.4f}")

# Save pre-trained model F
torch.save(model.state_dict(), "model_f.pth")
print("Model F pre-trained and saved.")