import torch
import torch.nn as nn
from typing import Optional


def get_embeddings(
    file: Optional[str] = None,
    freeze: Optional[str] = True,
    vocab_size: Optional[int] = None,
    embedding_dim: Optional[int] = None,
    **kwargs
):
    if file is None:
        if vocab_size is None:
            return
        embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)
    else:
        embeddings = nn.Embedding.from_pretrained(
            embeddings=torch.load(file),
            freeze=freeze)
    return embeddings