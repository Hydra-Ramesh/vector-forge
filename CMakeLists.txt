cmake_minimum_required(VERSION 3.20)
project(VectorForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Hyper-Optimization Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mavx2 -mfma -ffast-math")

# Enable OpenMP if available
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(include)

# Core & Index library files
set(VECTORFORGE_SOURCES
    src/core/math.cpp
    src/core/pq.cpp
    src/core/sys_utils.cpp
    src/core/dataset.cpp
    src/core/mmap_reader.cpp
    src/index/brute_force_index.cpp
    src/index/ivf_index.cpp
    src/index/ivfpq_index.cpp
    src/index/hnsw_index.cpp
    src/index/vamana_index.cpp
)

add_library(vectorforge_lib STATIC ${VECTORFORGE_SOURCES})
target_include_directories(vectorforge_lib PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vectorforge_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# CLI Executable
add_executable(vectorforge src/cli/main.cpp)
target_link_libraries(vectorforge PRIVATE vectorforge_lib)

# Tests
enable_testing()
add_subdirectory(tests)
