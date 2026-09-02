# VectorForge

A C++20 local-first vector database designed for high performance and learning. 
Supports Brute-Force and (planned) IVF-PQ approximate nearest-neighbor search.

## Build Requirements
- CMake 3.20+
- A C++20 compiler with OpenMP support (MSVC, GCC, Clang)

## Build Instructions (Windows/Linux)

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Running Tests

```bash
cd build
ctest -C Release --output-on-failure
```
