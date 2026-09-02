#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <new>
#if defined(_WIN32)
#include <malloc.h>
#endif

// Prefetch Macro
#if defined(__GNUC__) || defined(__clang__)
#define VFORGE_PREFETCH(ptr, rw, locality) __builtin_prefetch(ptr, rw, locality)
#elif defined(_MSC_VER)
#include <mmintrin.h>
#define VFORGE_PREFETCH(ptr, rw, locality) _mm_prefetch(reinterpret_cast<const char *>(ptr), _MM_HINT_T0)
#else
#define VFORGE_PREFETCH(ptr, rw, locality)
#endif

namespace vectorforge
{

    // Aligned Allocator for SIMD (32-byte boundary)
    template <typename T, std::size_t Alignment = 32>
    struct AlignedAllocator
    {
        using value_type = T;

        AlignedAllocator() noexcept = default;
        template <typename U>
        AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

        T *allocate(std::size_t n)
        {
            if (n == 0)
                return nullptr;
            void *ptr = nullptr;
            std::size_t bytes = n * sizeof(T);
#if defined(_WIN32)
            ptr = __mingw_aligned_malloc(bytes, Alignment);
            if (!ptr)
                ptr = _aligned_malloc(bytes, Alignment);
            if (!ptr)
                throw std::bad_alloc();
#else
            if (posix_memalign(&ptr, Alignment, bytes) != 0)
                throw std::bad_alloc();
#endif
            return static_cast<T *>(ptr);
        }

        void deallocate(T *p, std::size_t) noexcept
        {
#if defined(_WIN32)
            __mingw_aligned_free(p);
            // _aligned_free(p);
#else
            free(p);
#endif
        }

        template <typename U>
        struct rebind
        {
            using other = AlignedAllocator<U, Alignment>;
        };
    };

    template <typename T, typename U, std::size_t Alignment>
    bool operator==(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, Alignment> &) { return true; }
    template <typename T, typename U, std::size_t Alignment>
    bool operator!=(const AlignedAllocator<T, Alignment> &, const AlignedAllocator<U, Alignment> &) { return false; }

    using AlignedFloatVector = std::vector<float, AlignedAllocator<float>>;

    class MemoryProfiler
    {
    public:
        static double get_current_rss_mb();
        static double get_peak_rss_mb();
    };

    class Timer
    {
    public:
        Timer();
        void reset();
        double elapsed_ms() const;

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    };

} // namespace vectorforge
