import numpy as np
from typing import List
from app.config import DEFAULT_EMBED_MODEL

_embedding_model = None

def get_embedding_model():
    """Lazily load FastEmbed ONNX model."""
    global _embedding_model
    if _embedding_model is None:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=DEFAULT_EMBED_MODEL)
    return _embedding_model

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Generates dense embeddings for a list of strings and normalizes to unit length (L2 norm).
    Pre-normalization allows cosine similarity to be computed via pure dot product.
    """
    model = get_embedding_model()
    embeddings = list(model.embed(texts))
    matrix = np.array(embeddings, dtype=np.float32)
    
    # Normalize each vector so ||v|| = 1
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return matrix / norms

def embed_query(query: str) -> np.ndarray:
    """Embeds a single query string into a normalized 1D vector."""
    model = get_embedding_model()
    embedding = list(model.embed([query]))[0]
    vec = np.array(embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return vec / (norm if norm > 0 else 1e-12)
