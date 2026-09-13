import os
print("Starting script...")
if os.name == 'nt':
    print("Adding DLL directory...")
    os.add_dll_directory(r"C:\msys64\ucrt64\bin")
print("Importing vectorforge...")
import vectorforge
print("Import successful!")
import random

def main():
    print("Testing BruteForceIndex directly...")
    try:
        bf = vectorforge.BruteForceIndex(3)
        bf.add(1, [1.0, 0.0, 0.0])
        bf.build()
        opts = vectorforge.SearchOptions()
        bf.search([1.0, 0.0, 0.0], opts)
        print("BruteForceIndex ok")
    except Exception as e:
        print("BruteForceIndex failed:", e)
        
    print("Testing VamanaIndex directly...")
    try:
        vam = vectorforge.VamanaIndex(dimension=3, max_degree=4, candidate_list_size=10, pruning_alpha=1.2)
        vam.search([1.0, 0.0, 0.0], opts)
        print("VamanaIndex ok")
    except Exception as e:
        print("VamanaIndex failed:", e)

    print("Testing VectorForge DeltaIndex...")
    try:
        delta = vectorforge.DeltaIndex(dimension=3, max_degree=4, candidate_list_size=10, pruning_alpha=1.2)
        print("Created DeltaIndex")
        
        delta.add(1, [1.0, 0.0, 0.0])
        delta.add(2, [0.0, 1.0, 0.0])
        delta.add(3, [0.0, 0.0, 1.0])
        print("Added data to DeltaIndex")
        
        opts = vectorforge.SearchOptions()
        opts.top_k = 2
        
        print("Starting delta search...")
        results = delta.search([1.0, 0.1, 0.0], opts)
        print("Delta search finished!")
        print(f"Delta Search Results before merge: {[r.id for r in results]}")
        
        delta.merge()
        
        results = delta.search([1.0, 0.1, 0.0], opts)
        print(f"Delta Search Results after merge: {[r.id for r in results]}")
    except Exception as e:
        print(f"Error in DeltaIndex: {e}")
    
    print("\nTesting VectorForge HybridIndex...")
    try:
        hybrid = vectorforge.HybridIndex(dense_dim=3, max_degree=4, candidate_list_size=10, pruning_alpha=1.2)
        print("Created HybridIndex")
        
        hybrid.add(1, [1.0, 0.0, 0.0], {100: 1.5, 200: 0.5})
        hybrid.add(2, [0.0, 1.0, 0.0], {100: 0.5, 300: 1.0})
        print("Added data to HybridIndex")
        
        hybrid.build()
        print("Built HybridIndex")
        
        results = hybrid.search([0.9, 0.1, 0.0], {100: 1.0, 200: 1.0}, opts, 60.0)
        print(f"Hybrid Search Results (expected ID 1 to win): {[r.id for r in results]}")
        print("All tests passed!")
    except Exception as e:
        print(f"Error in HybridIndex: {e}")

if __name__ == "__main__":
    main()
