#include "vectorforge/core/collection_manager.hpp"

namespace vectorforge {

bool CollectionManager::create_collection(const std::string& name, size_t dimension) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (collections_.find(name) != collections_.end()) {
        return false;
    }
    // HybridIndex requires dense_dim. The default parameters handle the rest.
    collections_[name] = std::make_shared<HybridIndex>(dimension);
    return true;
}

std::shared_ptr<HybridIndex> CollectionManager::get_collection(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = collections_.find(name);
    if (it != collections_.end()) {
        return it->second;
    }
    return nullptr;
}

bool CollectionManager::delete_collection(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    return collections_.erase(name) > 0;
}

std::vector<std::string> CollectionManager::list_collections() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(collections_.size());
    for (const auto& pair : collections_) {
        names.push_back(pair.first);
    }
    return names;
}

} // namespace vectorforge
