"""Small debugger entry point for AudioFlow's original model component."""

import torch

from audio_flow.adaptors.onehot import OnehotEncoder


encoder = OnehotEncoder(num_classes=10, dim=32)
conditions = {"id": torch.tensor([0, 3, 9])}
embeddings = encoder(conditions)

assert embeddings["c"].shape == (3, 32)
print("Debugger smoke passed:", embeddings["c"].shape)
