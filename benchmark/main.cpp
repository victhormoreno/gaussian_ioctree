#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>

#include "map_benchmark.hpp"

int main(int argc, char** argv) {
    std::cout << "[Setup] Starting Map Benchmarking Instance Suite..." << std::endl;

    // Config defaults
    double resolution = 2.0;
    size_t update_threshold = 5;
    size_t points_per_frame = 25000;
    size_t TotalFrames = 200; 
    std::string pcd_path = "";

    // Quick Command-Line Check
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--pcd" && i + 1 < argc) pcd_path = argv[++i];
        if (arg == "--res" && i + 1 < argc) resolution = std::stod(argv[++i]);
        if (arg == "--thresh" && i + 1 < argc) update_threshold = std::stoull(argv[++i]);
        if (arg == "--frames" && i + 1 < argc) TotalFrames = std::stoull(argv[++i]);
    }

    size_t rss_base = benchmark::getProcessRSS_KB();

    // 1. Instantiating Selected Target Layout
    std::unique_ptr<benchmark::IBenchmarkableMap> test_map = 
        std::make_unique<benchmark::GenericWrapper<gauss_ivox_mapping::GaussianIVox>>(resolution, update_threshold);

    benchmark::BenchResult report;
    report.test_name = "Bonxai-GaussianIVox Engine (Pure C++)";
    report.process_rss_kb_start = rss_base;

    std::vector<gauss_ivox_mapping::pointWithCov> full_cloud;
    if (!pcd_path.empty()) {
        std::cout << "[Dataset] Loading physical PCD frames from: " << pcd_path << std::endl;
        full_cloud = loadPointsFromPCD(pcd_path);
        if (full_cloud.empty()) return -1;
        points_per_frame = full_cloud.size() / TotalFrames;
        if (points_per_frame == 0) points_per_frame = full_cloud.size();
    } else {
        std::cout << "[Dataset] Generating synthetic spiral data array segments..." << std::endl;
        full_cloud = generateSyntheticCloud(points_per_frame * TotalFrames, 50.0);
    }

    std::cout << "[Execute] Simulating " << TotalFrames << " sequential LiDAR sweeps (" 
              << points_per_frame << " points/sweep)..." << std::endl;

    size_t peak_rss = rss_base;
    double accumulated_ms = 0.0;

    for (size_t f = 0; f < TotalFrames; ++f) {
        // Step out next dataset segment window
        size_t start_idx = f * points_per_frame;
        if (start_idx >= full_cloud.size()) break;
        size_t end_idx = std::min(start_idx + points_per_frame, full_cloud.size());

        std::vector<gauss_ivox_mapping::pointWithCov> frame_packet(
            full_cloud.begin() + start_idx, full_cloud.begin() + end_idx
        );

        // --- Run Core Update Pipeline Measurement ---
        auto tick = std::chrono::high_resolution_clock::now();
        
        test_map->insertBatch(frame_packet);
        
        auto tack = std::chrono::high_resolution_clock::now();
        // ---------------------------------------------

        double frame_ms = std::chrono::duration<double, std::milli>(tack - tick).count();
        accumulated_ms += frame_ms;
        report.max_time_single_frame_ms = std::max(report.max_time_single_frame_ms, frame_ms);
        report.total_inserted_points += frame_packet.size();

        // Track continuous kernel virtual allocation peak increments 
        size_t current_rss = benchmark::getProcessRSS_KB();
        peak_rss = std::max(peak_rss, current_rss);
    }

    report.total_time_ms = accumulated_ms;
    report.avg_time_per_frame_ms = accumulated_ms / static_cast<double>(TotalFrames);
    report.active_voxels = test_map->getVoxelCount();
    report.map_internal_memory_bytes = test_map->getInternalMemory();
    report.process_rss_kb_peak = peak_rss;

    // Output Report Metrics
    printReport(report);

    // Save Metrics to CSV for automated tracking
    std::ofstream csv("benchmark_results.csv", std::ios::app);
    csv << report.test_name << "," << resolution << "," << update_threshold << ","
        << report.total_inserted_points << "," << report.active_voxels << ","
        << report.avg_time_per_frame_ms << "," << report.max_time_single_frame_ms << ","
        << static_cast<double>(report.map_internal_memory_bytes) / (1024.0 * 1024.0) << ","
        << static_cast<double>(report.process_rss_kb_peak) / 1024.0 << "\n";
    csv.close();

    return 0;
}