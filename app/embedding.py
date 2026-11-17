"""Embedding model wrapper (fastembed / ONNX, so no PyTorch needed)."""
from typing import Callable

import numpy as np

from . import config

EmbedFn = Callable[[list[str]], np.ndarray]


def get_embedder() -> EmbedFn:
    """Load the model once and return embed(texts) -> L2-normalised float32 matrix."""
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name=config.EMBED_MODEL, cache_dir=str(config.EMBED_CACHE_DIR))

    def embed(texts: list[str]) -> np.ndarray:
        vectors = np.array(list(model.embed(texts, batch_size=32)), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms  # unit length, so inner product == cosine similarity

    return embed
