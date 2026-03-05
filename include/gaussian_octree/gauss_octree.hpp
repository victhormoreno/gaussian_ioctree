#pragma once

#include "gaussian_octree/gauss_common.hpp"
#include <queue>

namespace gauss_mapping {

// Morton-order based neighbor lookup table for optimized traversal
static const int ordered_indices[8][7] = {
    {1, 2, 4, 3, 5, 6, 7}, {0, 3, 5, 2, 4, 7, 6}, {3, 0, 6, 1, 7, 4, 5}, {2, 1, 7, 0, 6, 5, 4},
    {5, 6, 0, 7, 1, 2, 3}, {4, 7, 1, 6, 0, 3, 2}, {7, 4, 2, 5, 3, 0, 1}, {6, 5, 3, 4, 2, 1, 0}
};

class Octree {
public:
    Octree(const Point& center, Scalar extent, size_t max_g = 5, Scalar min_e = 0.1, Scalar chi_threshold = 11.345)
        : num_points_(0), max_gaussians_(max_g), min_extent_(min_e), chi_threshold_(chi_threshold) {
        root_ = std::make_unique<Octant>(center, extent);
    }

    void update(const VecPointCov& points, const Mat3& P_curr) {
        updateRecursive(root_.get(), points, P_curr);
    }

    size_t size() { return num_points_; }

    void radiusSearch(Octant* node, const Point& q, Scalar r2, GaussianVector& results) const {
        if (node->isLeaf()) {
            for (const auto& g : node->gaussians) {
                if ((g.mean - q).squaredNorm() < r2) results.push_back(g);
            }
        } else {
            for (int i = 0; i < 8; ++i) {
                if (boxOverlap(node->children[i], q, r2))
                    radiusSearch(&node->children[i], q, r2, results);
            }
        }
    }

    template <typename PointT>
    void knnSearch(const PointT& query, int k, std::vector<PointT> &neighbors, std::vector<float> &distances) const {
        NeighborHeap heap(k);
        Point q = Point(query.x, query.y, query.z);
        knnSearch(q, heap);

        std::vector<Neighbor> points = heap.data();
        neighbors.reserve(points.size());
        distances.reserve(points.size());

        for(auto& p : points){
            PointT pt;
            pt.x = p.g.mean.x();
            pt.y = p.g.mean.y();
            pt.z = p.g.mean.z();
            pt.intensity = p.g.R_map.sum();
            neighbors.push_back(pt);
            distances.push_back(std::sqrt(p.dist_sq));
        }

    }

    template <typename PointT>
    NeighborHeap knnSearch(const PointT& query, int k) const {
        NeighborHeap heap(k);
        Point q = Point(query.x, query.y, query.z);
        knnSearch(q, heap);
        return heap;
    }

    void knnSearch(const Point& q, NeighborHeap& heap) const {
        if (!root_) return;
        knnRecursive(root_.get(), q, heap);
    }

    GaussianVector getGaussians() {
        GaussianVector gauss;
        getAllGaussians(root_.get(), gauss);

        return gauss;
    }

    template <typename PointT, typename ContainerT>
    ContainerT getData() {
        GaussianVector points;
        getAllGaussians(root_.get(), points);

        ContainerT out;
        out.reserve(points.size());

        for (const auto& p : points) {
            PointT pt;
            pt.x = p.mean.x();
            pt.y = p.mean.y();
            pt.z = p.mean.z();
            pt.intensity = p.R_map.trace();
            out.push_back(pt);
        }

        return out;
    }

protected:
    std::unique_ptr<Octant> root_;
    size_t num_points_;    // number of points in the tree
    size_t max_gaussians_; // maximum number of points allowed in an octant before it gets subdivided
    Scalar min_extent_;    // minimum extent of the octant (used to stop subdividing)
    Scalar chi_threshold_; // Mahalanobis distance threshold for fitting points to Gaussians

    void updateRecursive(Octant* octant, const VecPointCov& points, const Mat3& P_curr) {
        if (octant->isLeaf()) {
            // Add points to existing Gaussians or create new ones
            for (const auto& pt : points) {
                bool fused = false;

                // Check fit with existing Gaussians
                for (auto& g : octant->gaussians) {
                    if (g.checkFit(pt.pos, pt.cov, chi_threshold_)) {
                        g.fuseAdaptive(pt, P_curr);
                        fused = true;
                        break;
                    }
                }

                // If no fusion, create new Gaussian
                if (!fused) {
                    octant->gaussians.emplace_back(pt);
                    num_points_++;
                }
            }

            // Merge overlapping Gaussians after the whole batch is in
            mergeGaussiansInOctant(octant);

            // Split if necessary
            if (octant->gaussians.size() > max_gaussians_ && octant->extent > min_extent_) {
                split(octant);
            }
        } else {
            // Internal Node logic: Batch distribute points to children
            std::vector<VecPointCov> child_buckets(8);
            for (const auto& pt : points) {
                child_buckets[getIdx(pt.pos, octant->centroid)].push_back(pt);
            }

            #pragma omp parallel for
            for (int i = 0; i < 8; ++i) {
                if (child_buckets[i].empty()) continue;

                #pragma omp critical
                {
                    if (!octant->children) octant->init_children(); // Ensure child exists
                }

                updateRecursive(&octant->children[i], child_buckets[i], P_curr);
            }
        }
    }

    void split(Octant* octant) {
        octant->init_children();
        
        auto old_gaussians = std::move(octant->gaussians);
        octant->gaussians.clear();

        for (auto& g : old_gaussians) {
            // Re-wrap Gaussian into PointCov to re-insert
                // Note: We don't want to re-fuse and change R_map, just re-locate
            int idx = getIdx(g.mean, octant->centroid);
            octant->children[idx].gaussians.push_back(std::move(g));
        }
    }

    void mergeGaussiansInOctant(Octant* octant) {
        if (octant->gaussians.size() < 2) return;

        for (size_t i = 0; i < octant->gaussians.size(); ++i) {
            for (size_t j = i + 1; j < octant->gaussians.size(); ) {
                // Check if Gaussian J fits inside Gaussian I
                // We use the combined covariance S = Cov_i + Cov_j
                if (octant->gaussians[i].checkFit(octant->gaussians[j].mean, 
                                                    octant->gaussians[j].cov, 
                                                    chi_threshold_)) 
                {
                    // Perform the Weighted Bayesian Merge
                    octant->gaussians[i].merge(octant->gaussians[j]);
                    
                    // Remove the redundant Gaussian J
                    octant->gaussians.erase(octant->gaussians.begin() + j);
                } else {
                    ++j;
                }
            }
        }
    }

    static inline int getIdx(const Point& p, const Point& c) {
        return (p.x() > c.x() ? 1 : 0) | (p.y() > c.y() ? 2 : 0) | (p.z() > c.z() ? 4 : 0);
    }

    static bool boxOverlap(const Octant& node, const Point& q, Scalar r2) {
        Scalar d2 = 0;
        for(int i=0; i<3; ++i) {
            Scalar min_b = node.centroid[i] - node.extent, max_b = node.centroid[i] + node.extent;
            if (q[i] < min_b) d2 += std::pow(min_b - q[i], 2);
            else if (q[i] > max_b) d2 += std::pow(q[i] - max_b, 2);
        }
        return d2 <= r2;
    }

    bool isQuerySphereInside(Octant* octant, const Point& q, Scalar r2) const {
        // Distance from query to each face of the octant
        Point dists = octant->extent - (q - octant->centroid).cwiseAbs().array();
        
        // If query is outside or the sphere overlaps a boundary, return false
        if (dists.x() < 0 || dists.x() * dists.x() < r2) return false;
        if (dists.y() < 0 || dists.y() * dists.y() < r2) return false;
        if (dists.z() < 0 || dists.z() * dists.z() < r2) return false;
        
        return true;
    }

    bool knnRecursive(Octant* octant, const Point& q, NeighborHeap& heap) const {
        if (octant->isLeaf()) {
            for (const auto& g : octant->gaussians) {
                Scalar d2 = (q - g.mean).squaredNorm();
                heap.add(g, d2);
            }
            return heap.full() && isQuerySphereInside(octant, q, heap.worstDist());
        }

        // 1. Visit the child containing the query point first
        int morton = getIdx(q, octant->centroid);
        if (octant->children[morton].centroid.size() > 0) { // Check if valid
             if (knnRecursive(&octant->children[morton], q, heap)) return true;
        }

        // 2. Visit other children in optimized order
        for (int i = 0; i < 7; ++i) {
            int c = ordered_indices[morton][i];
            Octant* child = &octant->children[c];

            // Pruning: skip if child cannot possibly contain a closer point
            if (heap.full() && !overlaps(child, q, heap.worstDist()))
                continue;

            if (knnRecursive(child, q, heap)) return true;
        }

        return heap.full() && isQuerySphereInside(octant, q, heap.worstDist());
    }

    static Scalar minDistSq(Octant* node, const Point& q) {
        Scalar d2 = 0;
        for(int i=0; i<3; ++i) {
            Scalar min_b = node->centroid[i] - node->extent, max_b = node->centroid[i] + node->extent;
            if (q[i] < min_b) d2 += std::pow(min_b - q[i], 2);
            else if (q[i] > max_b) d2 += std::pow(q[i] - max_b, 2);
        }
        return d2;
    }

    static bool overlaps(Octant* node, const Point& q, Scalar r2) {
        return minDistSq(node, q) <= r2;
    }

    void getAllGaussians(Octant* node, GaussianVector& out_points) const {
        if (node == nullptr)
            return;

        if (node->isLeaf()) {
            out_points.insert(out_points.end(), node->gaussians.begin(), node->gaussians.end());
        } else if (node->children) {
            for (int i = 0; i < 8; ++i) {
                getAllGaussians(&node->children[i], out_points);
            }
        }
    }
    
    // Friend class to allow IVox to access root_
    friend class GaussianIVox;
};

} // namespace gauss_mapping
