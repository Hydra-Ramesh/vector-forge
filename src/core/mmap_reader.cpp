#include "vectorforge/core/mmap_reader.hpp"
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace vectorforge {

MmapReader::MmapReader() : data_(nullptr), size_(0) {
#ifdef _WIN32
    file_handle_ = (void*)-1; // INVALID_HANDLE_VALUE
    map_handle_ = nullptr;
#else
    fd_ = -1;
#endif
}

MmapReader::~MmapReader() {
    close();
}

MmapReader::MmapReader(MmapReader&& other) noexcept 
    : data_(other.data_), size_(other.size_) {
#ifdef _WIN32
    file_handle_ = other.file_handle_;
    map_handle_ = other.map_handle_;
    other.file_handle_ = (void*)-1;
    other.map_handle_ = nullptr;
#else
    fd_ = other.fd_;
    other.fd_ = -1;
#endif
    other.data_ = nullptr;
    other.size_ = 0;
}

MmapReader& MmapReader::operator=(MmapReader&& other) noexcept {
    if (this != &other) {
        close();
        data_ = other.data_;
        size_ = other.size_;
#ifdef _WIN32
        file_handle_ = other.file_handle_;
        map_handle_ = other.map_handle_;
        other.file_handle_ = (void*)-1;
        other.map_handle_ = nullptr;
#else
        fd_ = other.fd_;
        other.fd_ = -1;
#endif
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void MmapReader::open(const std::string& path) {
    close();

#ifdef _WIN32
    file_handle_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (file_handle_ == (void*)-1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle_, &file_size)) {
        CloseHandle(file_handle_);
        file_handle_ = (void*)-1;
        throw std::runtime_error("Failed to get file size: " + path);
    }
    size_ = static_cast<size_t>(file_size.QuadPart);

    if (size_ == 0) {
        return;
    }

    map_handle_ = CreateFileMappingA(file_handle_, NULL, PAGE_READONLY, 0, 0, NULL);
    if (map_handle_ == nullptr) {
        CloseHandle(file_handle_);
        file_handle_ = (void*)-1;
        throw std::runtime_error("Failed to create file mapping: " + path);
    }

    data_ = static_cast<uint8_t*>(MapViewOfFile(map_handle_, FILE_MAP_READ, 0, 0, 0));
    if (data_ == nullptr) {
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = nullptr;
        file_handle_ = (void*)-1;
        throw std::runtime_error("Failed to map view of file: " + path);
    }
#else
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Failed to open file: " + path);
    }

    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to get file size: " + path);
    }
    size_ = sb.st_size;

    if (size_ == 0) {
        return;
    }

    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        throw std::runtime_error("Failed to map file: " + path);
    }
    data_ = static_cast<uint8_t*>(mapped);
#endif
}

void MmapReader::close() {
    if (data_ != nullptr) {
#ifdef _WIN32
        UnmapViewOfFile(data_);
        CloseHandle(map_handle_);
        CloseHandle(file_handle_);
        map_handle_ = nullptr;
        file_handle_ = (void*)-1;
#else
        ::munmap(data_, size_);
        ::close(fd_);
        fd_ = -1;
#endif
        data_ = nullptr;
        size_ = 0;
    }
}

} // namespace vectorforge
