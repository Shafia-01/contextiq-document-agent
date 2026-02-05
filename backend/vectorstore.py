import numpy as np
from typing import List, Dict

class InMemoryVectorStore:
    def __init__(self):
        self.vectors = []    
        self.metadatas = []  
        self.ids = []        

    def add(self, vec: List[float], metadata: Dict, id: str = None):
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec) + 1e-10
        vec = vec / norm
        self.vectors.append(vec)
        self.metadatas.append(metadata)
        self.ids.append(id or str(len(self.ids)))

    def similarity_search(self, query_vec: List[float], top_k: int = 5):
        if len(self.vectors) == 0:
            return []
        q = np.array(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        mat = np.stack(self.vectors, axis=0) 
        scores = mat.dot(q)
        idx = np.argsort(-scores)[:top_k]
        results = []
        for i in idx:
            results.append({"id": self.ids[i], "score": float(scores[i]), "metadata": self.metadatas[i]})
        return results

    def __len__(self):
        return len(self.vectors)
