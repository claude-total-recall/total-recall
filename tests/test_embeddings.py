"""Tests for embedding operations."""

import pytest

from total_recall import embeddings as emb


class TestEmbeddings:
    def test_generate_embedding(self):
        embedding = emb.generate_embedding("Hello world")
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

    def test_cosine_similarity_identical(self):
        vec = [1.0, 0.0, 0.0]
        sim = emb.cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.0001

    def test_cosine_similarity_orthogonal(self):
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        sim = emb.cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.0001

    def test_search_embeddings(self):
        query_emb = emb.generate_embedding("programming language")
        embeddings = [
            ("python", emb.generate_embedding("Python is a programming language")),
            ("recipe", emb.generate_embedding("How to bake a cake")),
            ("java", emb.generate_embedding("Java programming tutorial")),
        ]

        results = emb.search_embeddings(query_emb, embeddings, limit=2, threshold=0.0)
        assert len(results) == 2
        # Programming-related should rank higher
        keys = [r[0] for r in results]
        assert "recipe" not in keys or keys.index("recipe") > 0

    def test_search_with_threshold(self):
        query_emb = emb.generate_embedding("cats and dogs")
        embeddings = [
            ("pets", emb.generate_embedding("I love my pet cat")),
            ("math", emb.generate_embedding("Calculus and algebra")),
        ]

        results = emb.search_embeddings(query_emb, embeddings, threshold=0.5)
        # Math should likely be filtered out
        assert all(score >= 0.5 for _, score in results)
