"""Vector embedding logic for Total Recall."""

import os
from typing import Optional

import numpy as np

# Lazy load sentence-transformers to avoid slow import on startup
_model = None
_model_name = None


def get_model_name() -> str:
    """Get the configured embedding model name."""
    return os.environ.get("TOTAL_RECALL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_model():
    """Get or initialize the sentence transformer model."""
    global _model, _model_name

    model_name = get_model_name()

    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)
        _model_name = model_name

    return _model


def generate_embedding(text: str) -> Optional[list[float]]:
    """Generate embedding for text.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding, or None on failure
    """
    try:
        model = get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)

    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


def search_embeddings(
    query_embedding: list[float],
    embeddings: list[tuple[str, list[float]]],
    limit: int = 10,
    threshold: float = 0.3,
) -> list[tuple[str, float]]:
    """Search embeddings by similarity to query.

    Args:
        query_embedding: The query vector
        embeddings: List of (key, embedding) tuples
        limit: Maximum results to return
        threshold: Minimum similarity score

    Returns:
        List of (key, score) tuples, sorted by score descending
    """
    results = []

    for key, embedding in embeddings:
        score = cosine_similarity(query_embedding, embedding)
        if score >= threshold:
            results.append((key, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:limit]
