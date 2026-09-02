#pragma once

#include <string>
#include <cstdint>
#include <cstddef>

namespace vectorforge {

class MmapReader {
public:
    MmapReader();
    ~MmapReader();

    // Disable copy
    MmapReader(const MmapReader&) = delete;
    MmapReader& operator=(const MmapReader&) = delete;

    // Enable move
    MmapReader(MmapReader&& other) noexcept;
    MmapReader& operator=(MmapReader&& other) noexcept;

    void open(const std::string& path);
    void close();

    const uint8_t* data() const { return data_; }
    size_t size() const { return size_; }
    bool is_open() const { return data_ != nullptr; }

private:
    uint8_t* data_;
    size_t size_;

#ifdef _WIN32
    void* file_handle_;
    void* map_handle_;
#else
    int fd_;
#endif
};

} // namespace vectorforge
