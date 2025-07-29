project/
│
├── graph_construction.py       # Dynamic graph construction
├── sthaf_encoder.py            # STHAF Encoder implementation
├── contrastive_loss.py         # Spatio-temporal contrastive loss
├── augmentations.py            # Data augmentation functions
├── main.py                     # Training pipeline and execution
└── requirements.txt            # Dependencies


Dependencies: Install via pip install -r requirements.txt.
Data: Replace the dummy np.random.rand with your real multivariate time-series dataset.
Scalability: Adjust batching and memory handling for larger datasets.
Fine-Tuning: Add a separate script or function in main.py for cross-domain fine-tuning with a classification head if needed.

Run main.py to pre-train model F. 
