import os
import requests
from typing import List, Dict, Optional

class VectorForgeClient:
    """
    A Python client to interact with the VectorForge Managed Cloud Service (Tier 3)
    or Self-Hosted Docker Server (Tier 2).
    """
    def __init__(self, url: str = "http://localhost:8000", api_key: str = None):
        self.url = url.rstrip('/')
        self.api_key = api_key or os.getenv("VECTORFORGE_API_KEY", "sk-vectorforge-dev")
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def create_collection(self, name: str, dimension: int):
        payload = {"name": name, "dimension": dimension}
        resp = requests.post(f"{self.url}/collections/create", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to create collection: {resp.text}")
        return resp.json()

    def upsert(self, collection_name: str, vector_id: int, dense_vector: List[float], sparse_vector: Optional[Dict[int, float]] = None, metadata_mask: int = 0):
        payload = {
            "id": vector_id,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
            "metadata_mask": metadata_mask
        }
        resp = requests.post(f"{self.url}/collections/{collection_name}/upsert", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to upsert vector: {resp.text}")
        return resp.json()

    def search(self, collection_name: str, dense_query: List[float], sparse_query: Optional[Dict[int, float]] = None, top_k: int = 10, metadata_mask: int = 0):
        payload = {
            "dense_query": dense_query,
            "sparse_query": sparse_query,
            "top_k": top_k,
            "metadata_mask": metadata_mask
        }
        resp = requests.post(f"{self.url}/collections/{collection_name}/search", json=payload, headers=self.headers)
        if resp.status_code != 200:
            raise Exception(f"Failed to search: {resp.text}")
        return resp.json()
