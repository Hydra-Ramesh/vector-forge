import os
os.add_dll_directory(r"C:\msys64\ucrt64\bin")
import vectorforge
import time

def test_vamana():
    print("Testing VamanaIndex...")
    dim = 128
    index = vectorforge.VamanaIndex(dim, max_degree=32)
    
    # Add vectors
    vector1 = [0.1] * dim
    vector2 = [0.2] * dim
    
    index.add(1, vector1)
    index.add(2, vector2)
    
    start = time.time()
    index.build()
    print(f"Built index in {time.time() - start:.4f}s")
    
    opts = vectorforge.SearchOptions()
    opts.top_k = 2
    
    results = index.search(vector1, opts)
    print(f"Search results for vector 1:")
    for res in results:
        print(f"  ID: {res.id}, Distance: {res.distance:.6f}")

if __name__ == "__main__":
    test_vamana()
