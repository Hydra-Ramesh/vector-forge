#include "vectorforge/core/sys_utils.hpp"

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <fstream>
#include <string>
#include <unistd.h>
#endif

namespace vectorforge {

double MemoryProfiler::get_current_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        return static_cast<double>(info.WorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    long rss = 0;
    std::ifstream stat_stream("/proc/self/statm", std::ios_base::in);
    if (stat_stream.good()) {
        long dummy;
        stat_stream >> dummy >> rss;
        stat_stream.close();
        // statm rss is in pages
        long page_size = sysconf(_SC_PAGE_SIZE);
        return static_cast<double>(rss * page_size) / (1024.0 * 1024.0);
    }
    return 0.0;
#endif
}

double MemoryProfiler::get_peak_rss_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        return static_cast<double>(info.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
    return 0.0;
#else
    struct rusage rusage;
    if (getrusage(RUSAGE_SELF, &rusage) == 0) {
        // ru_maxrss is typically in kilobytes on Linux
        return static_cast<double>(rusage.ru_maxrss) / 1024.0;
    }
    return 0.0;
#endif
}

Timer::Timer() {
    reset();
}

void Timer::reset() {
    start_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsed_ms() const {
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start_;
    return diff.count();
}

} // namespace vectorforge
