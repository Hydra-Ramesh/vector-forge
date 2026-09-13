#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "vectorforge/core/types.hpp"
#include "vectorforge/index/brute_force_index.hpp"
#include "vectorforge/index/hnsw_index.hpp"
#include "vectorforge/index/vamana_index.hpp"
#include "vectorforge/index/ivf_index.hpp"
#include "vectorforge/index/ivfpq_index.hpp"
#include "vectorforge/index/quantized_vamana_index.hpp"
#include "vectorforge/index/delta_index.hpp"
#include "vectorforge/index/sparse_index.hpp"
#include "vectorforge/index/hybrid_index.hpp"
#include "vectorforge/core/collection_manager.hpp"

namespace py = pybind11;
using namespace vectorforge;

PYBIND11_MODULE(vectorforge, m) {
    m.doc() = "VectorForge python bindings for high-performance vector search.";

    py::enum_<Metric>(m, "Metric")
        .value("L2", Metric::L2)
        .value("Cosine", Metric::Cosine)
        .export_values();

    py::class_<SearchResult>(m, "SearchResult")
        .def(py::init<>())
        .def_readwrite("id", &SearchResult::id)
        .def_readwrite("distance", &SearchResult::distance);

    py::class_<SearchOptions>(m, "SearchOptions")
        .def(py::init<>())
        .def_readwrite("top_k", &SearchOptions::top_k)
        .def_readwrite("metric", &SearchOptions::metric)
        .def_readwrite("filter_mask", &SearchOptions::filter_mask);

    py::class_<BruteForceIndex>(m, "BruteForceIndex")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("add", &BruteForceIndex::add, py::arg("id"), py::arg("vector"))
        .def("build", &BruteForceIndex::build)
        .def("search", &BruteForceIndex::search, py::arg("query"), py::arg("options"))
        .def("save", &BruteForceIndex::save, py::arg("path"))
        .def("load", &BruteForceIndex::load, py::arg("path"));

    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<size_t, int, int>(), py::arg("dimension"), py::arg("max_connections") = 16, py::arg("construction_expansion") = 100)
        .def("add", &HNSWIndex::add, py::arg("id"), py::arg("vector"))
        .def("build", &HNSWIndex::build)
        .def("search", &HNSWIndex::search, py::arg("query"), py::arg("options"))
        .def("save", &HNSWIndex::save, py::arg("path"))
        .def("load", &HNSWIndex::load, py::arg("path"));

    py::class_<VamanaIndex>(m, "VamanaIndex")
        .def(py::init<size_t, size_t, size_t, float>(), py::arg("dimension"), py::arg("max_degree") = 64, py::arg("candidate_list_size") = 100, py::arg("pruning_alpha") = 1.2f)
        .def("add", &VamanaIndex::add, py::arg("id"), py::arg("vector"), py::arg("mask") = 0)
        .def("remove", &VamanaIndex::remove, py::arg("id"))
        .def("compact", &VamanaIndex::compact)
        .def("build", &VamanaIndex::build)
        .def("search", &VamanaIndex::search, py::arg("query"), py::arg("opts"))
        .def("save", &VamanaIndex::save, py::arg("filepath"))
        .def("load", &VamanaIndex::load, py::arg("filepath"));

    py::class_<QuantizedVamanaIndex>(m, "QuantizedVamanaIndex")
        .def(py::init<size_t, size_t, size_t, size_t, float>(), py::arg("dimension"), py::arg("m"), py::arg("max_degree") = 64, py::arg("candidate_list_size") = 100, py::arg("pruning_alpha") = 1.2f)
        .def("train", &QuantizedVamanaIndex::train, py::arg("training_data"), py::arg("metric") = Metric::L2)
        .def("add", &QuantizedVamanaIndex::add, py::arg("id"), py::arg("vector"), py::arg("mask") = 0)
        .def("remove", &QuantizedVamanaIndex::remove, py::arg("id"))
        .def("compact", &QuantizedVamanaIndex::compact)
        .def("build", &QuantizedVamanaIndex::build)
        .def("search", &QuantizedVamanaIndex::search, py::arg("query"), py::arg("opts"));

    py::class_<DeltaIndex>(m, "DeltaIndex")
        .def(py::init<size_t, size_t, size_t, float>(), py::arg("dimension"), py::arg("max_degree") = 64, py::arg("candidate_list_size") = 100, py::arg("pruning_alpha") = 1.2f)
        .def("add", &DeltaIndex::add, py::arg("id"), py::arg("vector"))
        .def("remove", &DeltaIndex::remove, py::arg("id"))
        .def("merge", &DeltaIndex::merge)
        .def("search", &DeltaIndex::search, py::arg("query"), py::arg("opts"));

    py::class_<SparseIndex>(m, "SparseIndex")
        .def(py::init<>())
        .def("add", &SparseIndex::add, py::arg("id"), py::arg("sparse_vec"))
        .def("remove", &SparseIndex::remove, py::arg("id"))
        .def("search", &SparseIndex::search, py::arg("query"), py::arg("opts"));

    py::class_<HybridIndex>(m, "HybridIndex")
        .def(py::init<size_t, size_t, size_t, float>(), py::arg("dense_dim"), py::arg("max_degree") = 64, py::arg("candidate_list_size") = 100, py::arg("pruning_alpha") = 1.2f)
        .def("add", &HybridIndex::add, py::arg("id"), py::arg("dense_vec"), py::arg("sparse_vec"), py::arg("mask") = 0)
        .def("build", &HybridIndex::build)
        .def("search", &HybridIndex::search, py::arg("dense_query"), py::arg("sparse_query"), py::arg("opts"), py::arg("rrf_k") = 60.0f);

    py::class_<IVFIndex>(m, "IVFIndex")
        .def(py::init<size_t>(), py::arg("dimension"))
        .def("train", &IVFIndex::train, py::arg("data"), py::arg("nlist"), py::arg("metric") = Metric::L2)
        .def("add", &IVFIndex::add, py::arg("id"), py::arg("vector"))
        .def("build", &IVFIndex::build)
        .def("search", &IVFIndex::search, py::arg("query"), py::arg("options"), py::arg("nprobe") = 10)
        .def("save", &IVFIndex::save, py::arg("path"))
        .def("load", &IVFIndex::load, py::arg("path"));

    py::class_<IVFPQIndex>(m, "IVFPQIndex")
        .def(py::init<size_t, size_t, size_t, bool>(), py::arg("dimension"), py::arg("m"), py::arg("lut_bits") = 8, py::arg("use_simd") = true)
        .def("train", &IVFPQIndex::train, py::arg("data"), py::arg("nlist"), py::arg("metric") = Metric::L2)
        .def("add", &IVFPQIndex::add, py::arg("id"), py::arg("vector"))
        .def("build", &IVFPQIndex::build)
        .def("search", &IVFPQIndex::search, py::arg("query"), py::arg("options"), py::arg("nprobe") = 10, py::arg("rerank_n") = 50)
        .def("save", &IVFPQIndex::save, py::arg("path"))
        .def("load", &IVFPQIndex::load, py::arg("path"))
        .def("load_mmap", &IVFPQIndex::load_mmap, py::arg("path"));

    py::class_<CollectionManager>(m, "CollectionManager")
        .def(py::init<>())
        .def("create_collection", &CollectionManager::create_collection, py::arg("name"), py::arg("dimension"))
        .def("get_collection", &CollectionManager::get_collection, py::arg("name"))
        .def("delete_collection", &CollectionManager::delete_collection, py::arg("name"))
        .def("list_collections", &CollectionManager::list_collections);
}
