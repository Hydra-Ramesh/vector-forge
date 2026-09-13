cmake_minimum_required(VERSION 3.20)
project(VectorForge VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generic Optimization Flags
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /fp:fast")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math")
endif()

# Only enable AVX2 for math.cpp on Windows x86_64 (GCC/Clang use __attribute__ target)
if(MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "AMD64|amd64|x86_64|X86_64")
    set_source_files_properties(src/core/math.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
endif()

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
    src/core/fvecs_reader.cpp
    src/core/collection_manager.cpp
    src/index/brute_force_index.cpp
    src/index/ivf_index.cpp
    src/index/ivfpq_index.cpp
    src/index/hnsw_index.cpp
    src/index/vamana_index.cpp
    src/index/quantized_vamana_index.cpp
    src/index/delta_index.cpp
    src/index/sparse_index.cpp
    src/index/hybrid_index.cpp
)

add_library(vectorforge_lib STATIC ${VECTORFORGE_SOURCES})
target_include_directories(vectorforge_lib PUBLIC include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vectorforge_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

# CLI Executable
add_executable(vectorforge_cli src/cli/main.cpp)
target_link_libraries(vectorforge_cli PRIVATE vectorforge_lib)

# Python Bindings
if (BUILD_PYTHON_BINDINGS)
    include(FetchContent)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)

    pybind11_add_module(vectorforge src/python/bindings.cpp)
    target_link_libraries(vectorforge PRIVATE vectorforge_lib)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vectorforge PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# Tests
enable_testing()
add_subdirectory(tests)
