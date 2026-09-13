import os
import sys
import uuid
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel

# Add the parent directory to sys.path so we can import the locally built vectorforge module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# On Windows, if compiled with MinGW, we need the DLL directory
if os.name == 'nt':
    try:
        os.add_dll_directory(r"C:\msys64\ucrt64\bin")
    except Exception:
        pass

import vectorforge

app = FastAPI(title="VectorForge Enterprise Server", version="1.0.0")

# --- Security (Simple API Key) ---
API_KEY = os.getenv("VECTORFORGE_API_KEY", "sk-vectorforge-dev")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

# --- Native C++ Collection Manager ---
# The Multi-Tenancy Engine now lives entirely inside the C++ Core for maximum performance!
manager = vectorforge.CollectionManager()

# --- API Models ---
class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int

class UpsertRequest(BaseModel):
    id: int
    dense_vector: List[float]
    sparse_vector: Optional[Dict[int, float]] = None
    metadata_mask: Optional[int] = 0

class SearchRequest(BaseModel):
    dense_query: List[float]
    sparse_query: Optional[Dict[int, float]] = None
    top_k: int = 10
    metadata_mask: Optional[int] = 0

# --- Endpoints ---

@app.post("/collections/create")
def create_collection(req: CreateCollectionRequest, api_key: str = Depends(verify_api_key)):
    success = manager.create_collection(req.name, req.dimension)
    if not success:
        raise HTTPException(status_code=400, detail="Collection already exists")
    return {"status": "success", "message": f"Collection '{req.name}' created"}

@app.post("/collections/{collection_name}/upsert")
def upsert_vector(collection_name: str, req: UpsertRequest, api_key: str = Depends(verify_api_key)):
    hybrid = manager.get_collection(collection_name)
    if not hybrid:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    # We delegate directly to the C++ HybridIndex
    sparse_v = req.sparse_vector if req.sparse_vector else {}
    mask = req.metadata_mask if req.metadata_mask else 0
    hybrid.add(req.id, req.dense_vector, sparse_v, mask)
        
    return {"status": "success", "id": req.id}

@app.post("/collections/{collection_name}/search")
def search_vectors(collection_name: str, req: SearchRequest, api_key: str = Depends(verify_api_key)):
    hybrid = manager.get_collection(collection_name)
    if not hybrid:
        raise HTTPException(status_code=404, detail="Collection not found")
    
    opts = vectorforge.SearchOptions()
    opts.top_k = req.top_k
    opts.filter_mask = req.metadata_mask if req.metadata_mask else 0
    
    sparse_q = req.sparse_query if req.sparse_query else {}
    
    # Execute Native C++ RRF Hybrid Search
    results = hybrid.search(req.dense_query, sparse_q, opts)
    
    return {
        "status": "success",
        "results": [{"id": r.id, "distance": r.distance} for r in results]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
