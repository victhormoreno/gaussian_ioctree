#pragma once

#include "gaussian_octree/gauss_common.hpp"
#include <unordered_map>
#include <numeric>
#include <mutex>

namespace gauss_mapping {

class GaussianIVox {
public:

    struct HashVec3i {
        size_t operator()(const Eigen::Vector3i& v) const {
            return ((size_t)v[0] * 73856093) ^
                   ((size_t)v[1] * 19349663) ^
                   ((size_t)v[2] * 83492791);
        }
    };

    struct Voxel {
        std::vector<Gaussian> gaussians;
    };

    GaussianIVox(Scalar v_res = 1.0, size_t max_g = 5, Scalar chi = 11.345)
        : v_res_(v_res),
          inv_res_(1.0 / v_res),
          max_gaussians_per_voxel_(max_g),
          chi_threshold_(chi) { }

    /**
     * @brief Incremental update with local merging. 
     */
    void update(const VecPointCov& points, const Mat3& P_curr) {
        std::lock_guard<std::mutex> lock(map_mtx_);
        
        // 1. Group incoming points into temporal buckets
        std::unordered_map<Eigen::Vector3i, VecPointCov, HashVec3i> local_buckets;
        for (const auto& pt : points) {
            if (!pt.pos.allFinite()) continue;
            Eigen::Vector3i key = (pt.pos * inv_res_).array().floor().cast<int>();
            local_buckets[key].push_back(pt);
            total_points_++;
        }

        // 2. Process each affected voxel
        for (auto& [key, pts] : local_buckets) {
            auto& voxel = grids_[key];

            for (const auto& pt : pts) {
                bool fused = false;
                for (auto& g : voxel.gaussians) {
                    if (g.checkFit(pt.pos, pt.cov, chi_threshold_)) {
                        g.fuseAdaptive(pt, P_curr);
                        fused = true;
                        break;
                    }
                }

                if (!fused) {
                    Gaussian new_g(pt);
                    voxel.gaussians.push_back(new_g);
                    total_gaussians_++;
                }
            }

            // 3. Local Voxel Maintenance
            if (voxel.gaussians.size() > max_gaussians_per_voxel_) {
                mergeVoxelInternally(voxel);
            }
            
            // 4. Local Neighbor Merging (O(1) replacement for global merge)
            mergeWithNeighbors(key);
        }
    }

    /**
     * @brief Merges Gaussians in a voxel with those in the 26 adjacent voxels.
     */
    void mergeWithNeighbors(const Eigen::Vector3i& key) {
        auto it_main = grids_.find(key);
        if (it_main == grids_.end()) return;

        for (int x = -1; x <= 1; ++x) {
            for (int y = -1; y <= 1; ++y) {
                for (int z = -1; z <= 1; ++z) {
                    if (x == 0 && y == 0 && z == 0) continue;
                    
                    Eigen::Vector3i neighbor_key = key + Eigen::Vector3i(x, y, z);
                    auto it_neighbor = grids_.find(neighbor_key);
                    if (it_neighbor == grids_.end()) continue;

                    auto& main_gaussians = it_main->second.gaussians;
                    auto& neighbor_gaussians = it_neighbor->second.gaussians;

                    for (auto it_m = main_gaussians.begin(); it_m != main_gaussians.end(); ++it_m) {
                        for (auto it_n = neighbor_gaussians.begin(); it_n != neighbor_gaussians.end(); ) {
                            // Check spatial proximity and distribution fit
                            if ((it_m->mean - it_n->mean).norm() < v_res_ * 0.5 && 
                                 it_m->checkFit(it_n->mean, it_n->getCovariance(), chi_threshold_)) {
                                
                                it_m->merge(*it_n);
                                it_n = neighbor_gaussians.erase(it_n);
                                total_gaussians_--;
                            } else {
                                ++it_n;
                            }
                        }
                    }
                }
            }
        }
    }

    void mergeVoxelInternally(Voxel& voxel) {
        if (voxel.gaussians.empty()) return;
        for (size_t i = 0; i < voxel.gaussians.size(); ++i) {
            for (size_t j = i + 1; j < voxel.gaussians.size(); ) {
                if (voxel.gaussians[i].checkFit(voxel.gaussians[j].mean, 
                                               voxel.gaussians[j].cov, 
                                               chi_threshold_)) 
                {
                    voxel.gaussians[i].merge(voxel.gaussians[j]);
                    voxel.gaussians.erase(voxel.gaussians.begin() + j);
                    total_gaussians_--;
                } else {
                    ++j;
                }
            }
        }
    }

    void radiusSearch(const Point& q, Scalar r, 
                      std::vector<Gaussian*>& neighbors, 
                      std::vector<Scalar>& distances) {
        std::lock_guard<std::mutex> lock(map_mtx_);
        neighbors.clear();
        distances.clear();

        int r_v = std::ceil(r * inv_res_);
        Eigen::Vector3i ck = (q * inv_res_).array().floor().cast<int>();
        Scalar r2 = r * r;

        for (int x = -r_v; x <= r_v; x++) {
            for (int y = -r_v; y <= r_v; y++) {
                for (int z = -r_v; z <= r_v; z++) {
                    auto it = grids_.find(ck + Eigen::Vector3i(x, y, z));
                    if (it == grids_.end()) continue;

                    for (auto& g : it->second.gaussians) {
                        Scalar d2 = (g.mean - q).squaredNorm();
                        if (d2 <= r2) {
                            neighbors.push_back(&g);
                            distances.push_back(std::sqrt(d2));
                        }
                    }
                }
            }
        }
    }

    void clear() {
        std::lock_guard<std::mutex> lock(map_mtx_);
        grids_.clear();
    }

    /**
    * @brief Returns pointers to all Gaussians in the map.
    * @note The pointers are valid until the next 'update' or 'clear' call.
    */
    std::vector<Gaussian*> getGaussians() {
        std::lock_guard<std::mutex> lock(map_mtx_);
        
        std::vector<Gaussian*> all_gaussians;
        // Pre-allocate based on atomic counter for speed
        all_gaussians.reserve(total_gaussians_.load()); 

        for (auto& [key, voxel] : grids_) {
            for (auto& g : voxel.gaussians) {
                all_gaussians.push_back(&g);
            }
        }
        return all_gaussians;
    }

    /**
    * @brief Returns a deep copy of all Gaussians.
    * Safe to use even if the main map is updated later.
    */
    std::vector<Gaussian> getGaussiansCopy() const {
        std::lock_guard<std::mutex> lock(map_mtx_);
        
        std::vector<Gaussian> copies;
        copies.reserve(total_gaussians_.load());

        for (const auto& [key, voxel] : grids_) {
            for (const auto& g : voxel.gaussians) {
                copies.push_back(g);
            }
        }
        return copies;
    }

    /**
    * @brief Returns the total number of Gaussians currently in the map.
    * This represents the "compressed" size of the map.
    */
    size_t size() const {
        return total_gaussians_.load();
    }

    /**
    * @brief Returns the total number of raw points fused into the map.
    */
    size_t num_points() const {
        return total_points_.load();
    }

private:
    Scalar v_res_, inv_res_, chi_threshold_;

    size_t max_gaussians_per_voxel_;

    std::unordered_map<Eigen::Vector3i, Voxel, HashVec3i> grids_;
    mutable std::mutex map_mtx_; 

    std::atomic<size_t> total_gaussians_{0};
    std::atomic<size_t> total_points_{0};
};

} // namespace gauss_mapping