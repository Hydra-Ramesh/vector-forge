#pragma once
#include <string>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <vector>
#include "vectorforge/index/hybrid_index.hpp"

namespace vectorforge {

class CollectionManager {
private:
    std::unordered_map<std::string, std::shared_ptr<HybridIndex>> collections_;
    std::mutex mutex_;

public:
    CollectionManager() = default;
    
    // Create a new collection. Returns true if created, false if it already exists.
    bool create_collection(const std::string& name, size_t dimension);
    
    // Get a collection by name. Returns nullptr if not found.
    std::shared_ptr<HybridIndex> get_collection(const std::string& name);
    
    // Delete a collection.
    bool delete_collection(const std::string& name);
    
    // List all collection names.
    std::vector<std::string> list_collections();
};

} // namespace vectorforge
