// debug_timer.hpp
#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <unordered_map>
#include <mutex>
#include <iomanip>

#if ENABLE_PROFILING
class DebugScopedTimer {
    public:
        explicit DebugScopedTimer(const std::string& name)
            : name_(name),
              start_(std::chrono::high_resolution_clock::now()) {}

        ~DebugScopedTimer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            std::cout << "[PROFILE] " << name_ << " took " 
                      << std::fixed << std::setprecision(2) << dur.count() << " us\n";
        }

    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };

// Macro for debug-only RAII
#define DEBUG_PROFILE_SCOPE(name) DebugScopedTimer timer_##__LINE__(name)

#else
    #define DEBUG_PROFILE_SCOPE(name)
#endif