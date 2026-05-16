#pragma once

#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <unistd.h>

#include "gaussian_octree/gauss_bonxai_ivox.hpp"
// #include "gaussian_octree/gauss_ivox.hpp"

namespace benchmark {

struct BenchResult {
    std::string test_name;
    size_t total_inserted_points = 0;
    size_t active_voxels = 0;
    double total_time_ms = 0.0;
    double avg_time_per_frame_ms = 0.0;
    double max_time_single_frame_ms = 0.0;
    size_t map_internal_memory_bytes = 0;
    size_t process_rss_kb_start = 0;
    size_t process_rss_kb_peak = 0;
};

// Base Interface to swap map implementations transparently
class IBenchmarkableMap {
public:
    virtual ~IBenchmarkableMap() = default;
    virtual void insertBatch(const std::vector<gauss_ivox_mapping::pointWithCov>& points) = 0;
    virtual size_t getVoxelCount() const = 0;
    virtual size_t getInternalMemory() const = 0;
    virtual void clearMap() = 0;
};

// =========================================================================
// Generic Template Wrapper for Any Map Class
// =========================================================================
template <typename MapClass>
class GenericWrapper : public IBenchmarkableMap {
public:
    GenericWrapper(double res, size_t thresh) {
        map_ptr_ = std::make_unique<MapClass>(res, thresh);
    }

    void insertBatch(const std::vector<gauss_ivox_mapping::pointWithCov>& points) override {
        map_ptr_->update(points);
    }

    size_t getVoxelCount() const override { return map_ptr_->size(); }
    size_t getInternalMemory() const override { return map_ptr_->memory(); }
    void clearMap() override { map_ptr_->clear(); }

private:
    std::unique_ptr<MapClass> map_ptr_;
};

// --- Linux System Metrics Utility ---
inline size_t getProcessRSS_KB() {
    std::string ignore;
    std::ifstream stat_stream("/proc/self/stat", std::ios_base::in);
    if (!stat_stream.is_open()) return 0;

    // Field 24 in /proc/self/stat is the Resident Set Size (RSS) in pages
    for (int i = 0; i < 23; ++i) stat_stream >> ignore;
    size_t rss_pages;
    stat_stream >> rss_pages;
    
    return rss_pages * (sysconf(_SC_PAGESIZE) / 1024);
}

} // namespace benchmark

// Optional PCD parsing manually to avoid heavy external PCL build dependencies if desired, 
// but here customized to fit standard ASCII/Binary PCD patterns directly.
inline std::vector<gauss_ivox_mapping::pointWithCov> loadPointsFromPCD(const std::string& path) {
    std::vector<gauss_ivox_mapping::pointWithCov> points;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[Error] Could not open PCD file: " << path << std::endl;
        return points;
    }

    std::string line;
    size_t num_points = 0;
    bool data_section = false;

    while (std::getline(file, line)) {
        if (!data_section) {
            if (line.rfind("POINTS", 0) == 0) {
                num_points = std::stoull(line.substr(7));
            }
            if (line.rfind("DATA", 0) == 0) {
                data_section = true;
                points.reserve(num_points);
            }
            continue;
        }
        
        if (line.empty()) continue;
        std::stringstream ss(line);
        double x, y, z;
        if (ss >> x >> y >> z) {
            gauss_ivox_mapping::pointWithCov pt;
            pt.p << x, y, z;
            pt.cov = 0.001 * gauss_ivox_mapping::Mat3::Identity(); // Regularization floor noise
            points.push_back(pt);
        }
    }
    return points;
}

// Generates an expanding spiral corridor pattern simulating real trajectory sweeps
inline std::vector<gauss_ivox_mapping::pointWithCov> generateSyntheticCloud(size_t num_points, double radius_scale) {
    std::vector<gauss_ivox_mapping::pointWithCov> points;
    points.reserve(num_points);

    std::mt19937 gen(42); // Fixed seed for matching conditions across iterations
    std::uniform_real_distribution<double> noise(-0.02, 0.02);

    for (size_t i = 0; i < num_points; ++i) {
        double t = static_cast<double>(i) / 1000.0;
        double x = radius_scale * std::cos(t) + noise(gen);
        double y = radius_scale * std::sin(t) + noise(gen);
        double z = t * 0.1 + noise(gen);

        gauss_ivox_mapping::pointWithCov pt;
        pt.p << x, y, z;
        pt.cov = 0.001 * gauss_ivox_mapping::Mat3::Identity();
        points.push_back(pt);
    }
    return points;
}

inline void printReport(const benchmark::BenchResult& res) {
    std::cout << "\n==================================================\n";
    std::cout << " BENCHMARK REPORT: " << res.test_name << "\n";
    std::cout << "==================================================\n";
    std::cout << " Total Points Pushed      : " << res.total_inserted_points << "\n";
    std::cout << " Instantiated Map Voxels  : " << res.active_voxels << "\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << " Accumulative Time        : " << std::fixed << std::setprecision(3) << res.total_time_ms << " ms\n";
    std::cout << " Avg Time per Frame       : " << res.avg_time_per_frame_ms << " ms\n";
    std::cout << " Worst Case Single Frame  : " << res.max_time_single_frame_ms << " ms\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << " Map Internal Memory      : " << std::fixed << std::setprecision(2) 
              << static_cast<double>(res.map_internal_memory_bytes) / (1024.0 * 1024.0) << " MB\n";
    std::cout << " System RSS (Initial OS)  : " << static_cast<double>(res.process_rss_kb_start) / 1024.0 << " MB\n";
    std::cout << " System RSS (Peak Process): " << static_cast<double>(res.process_rss_kb_peak) / 1024.0 << " MB\n";
    std::cout << "==================================================\n\n";
}