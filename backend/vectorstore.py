import numpy as np
from typing import List, Dict


class InMemoryVectorStore:
    """
    Minimal in-memory vector store with cosine similarity search.

    This intentionally keeps the retrieval layer transparent and easy to reason
    about for a junior engineer assessment:
    - Vectors are L2-normalised on write and on query.
    - Similarity is a simple dot product (cosine similarity because of
      normalisation).
    - Metadata is an arbitrary Python dict, typically containing:
        * the original chunk text
        * document-level attribution (names, paths, page numbers).

    In a production deployment this class can be replaced with FAISS, Pinecone,
    or Chroma while keeping the QA engine interface unchanged.
    """

    def __init__(self):
        self.vectors = []    # Normalised embedding vectors
        self.metadatas = []  # Arbitrary payloads per vector
        self.ids = []        # Stable ids (string)

    def add(self, vec: List[float], metadata: Dict, id: str = None):
        """Add a new vector and its associated metadata to the store."""
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm
        self.vectors.append(vec)
        self.metadatas.append(metadata)
        self.ids.append(id or str(len(self.ids)))

    def similarity_search(self, query_vec: List[float], top_k: int = 5):
        """
        Return the ``top_k`` most similar vectors to ``query_vec``.

        Results are sorted by decreasing cosine similarity and include:
        - ``id``: stored vector id
        - ``score``: cosine similarity in [-1, 1]
        - ``metadata``: original metadata supplied at ``add`` time.
        """
        if len(self.vectors) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        mat = np.stack(self.vectors, axis=0)
        scores = mat.dot(q)
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            results.append(
                {"id": self.ids[i], "score": float(scores[i]), "metadata": self.metadatas[i]}
            )
        return results

    def __len__(self):
        return len(self.vectors)
