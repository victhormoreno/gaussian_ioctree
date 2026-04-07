#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <memory>
#include <atomic>
#include <cmath>

namespace gauss_ivox_mapping {

using Scalar = double;
using Point  = Eigen::Matrix<Scalar, 3, 1>;
using Mat3   = Eigen::Matrix<Scalar, 3, 3>;

inline size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct HashVec3i {
    size_t operator()(const Eigen::Vector3i& v) const {
        size_t seed = 0;
        seed = hash_combine(seed, std::hash<int>()(v[0]));
        seed = hash_combine(seed, std::hash<int>()(v[1]));
        seed = hash_combine(seed, std::hash<int>()(v[2]));
        return seed;
    }
};

/*** 3D Point with Covariance ***/
struct pointWithCov
{
    Point point;
    Mat3 cov;
};

enum class PrimitiveType {
    UNKNOWN,
    LINE,
    PLANE,
    VOLUME
};

struct GaussianPrimitive
{
    // --- Core Gaussian ---
    Point mean = Point::Zero();
    Mat3  cov  = Mat3::Zero();

    // --- Classification ---
    PrimitiveType type = PrimitiveType::UNKNOWN;
    Point direction; // optional for LINE
    Point n_vec;     // optional for PLANE

    // --- Incremental stats ---
    Scalar xx = 0, yy = 0, zz = 0;
    Scalar xy = 0, xz = 0, yz = 0;
    Scalar x = 0, y = 0, z = 0;
    int count = 0;
};

using GaussPtr = std::shared_ptr<GaussianPrimitive>;
using GaussConstPtr = const std::shared_ptr<GaussianPrimitive>;

class UnionFindNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GaussPtr gauss_ptr;
    UnionFindNode *rootNode;
    
    // We store points until the first initGeometry(), then use incremental calculation.
    std::vector<pointWithCov> buffer_points; 
    
    bool init_node = false;
    Scalar voxel_center_[3];

    UnionFindNode(const Eigen::Vector3i& key, Scalar voxel_sz) {
        gauss_ptr = std::make_shared<GaussianPrimitive>();
        rootNode = this;
        voxel_center_[0] = (key.x() + 0.5) * voxel_sz;
        voxel_center_[1] = (key.y() + 0.5) * voxel_sz;
        voxel_center_[2] = (key.z() + 0.5) * voxel_sz;
    }

    // Path Compression for DSU
    UnionFindNode* find() {
        if (rootNode == this) return this;
        return rootNode = rootNode->find();
    }
};

class GaussianIVox {
public:
    GaussianIVox(Scalar v_sz, std::size_t upd_thresh) 
        : v_size_(v_sz), inv_v_size_(1.0 / v_sz), update_threshold_(upd_thresh) {}

    /**
     * @brief Updates the map with a new point. Handles DSU merging and geometry.
     */
    void update(const pointWithCov& pv) {
        Eigen::Vector3i key = (pv.point * inv_v_size_).array().floor().cast<int>();

        UnionFindNode* node = nullptr;
        {
            std::unique_lock<std::shared_mutex> lock(map_mtx_);
            auto it = map_.find(key);
            if (it == map_.end()) {
                node = new UnionFindNode(key, v_size_);
                map_[key] = node;
            } else {
                node = it->second;
            }
        }

        // Always operate on the root for DSU consistency
        UnionFindNode* root = node->find();
        root->buffer_points.push_back(pv);

        // Trigger geometry update if threshold reached
        if (root->buffer_points.size() >= update_threshold_) {
            if (!root->init_node) {
                initGeometry(root);
            } else {
                updateGeometry(root, key);
            }
            root->buffer_points.clear(); // Clear buffer to save memory
        }
    }

    /**
     * @brief Thread-safe retrieval of all unique gaussians in the map
     */
    std::vector<GaussPtr> getGaussians() const {
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        std::vector<GaussPtr> result;
        for (auto const& [_, node] : map_) {
            if (node->rootNode != node) continue; // only roots
            result.push_back(node->gauss_ptr);
        }
        return result;
    }

    /**
     * @brief Thread-safe retrieval of selected type gaussians
     */
    std::vector<GaussPtr> getByType(PrimitiveType type) const {
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        std::vector<GaussPtr> result;
        for (const auto& [_, node] : map_) {
            if (node->rootNode != node) continue; // only roots

            const auto& g = node->gauss_ptr;
            if (g && g->type == type) {
                result.push_back(g);
            }
        }

        return result;
    }

// private:
    /**
     * @brief Initial Gaussian Fitting and Eigen Decomposition
     */
    void initGeometry(UnionFindNode* node) {
        auto& g = node->gauss_ptr;

        // Incremental Summation (compute mean & scatter)
        for (const auto& pv : node->buffer_points) {
            g->count++;
            g->x  += pv.point[0]; g->y  += pv.point[1]; g->z  += pv.point[2];
            g->xx += pv.point[0]*pv.point[0]; g->yy += pv.point[1]*pv.point[1]; g->zz += pv.point[2]*pv.point[2];
            g->xy += pv.point[0]*pv.point[1]; g->xz += pv.point[0]*pv.point[2]; g->yz += pv.point[1]*pv.point[2];
        }

        Scalar n = static_cast<Scalar>(g->count);
        g->mean << g->x/n, g->y/n, g->z/n;

        // Covariance / scatter matrix
        g->cov << g->xx/n - (g->x/n)*(g->x/n), g->xy/n - (g->x/n)*(g->y/n), g->xz/n - (g->x/n)*(g->z/n),
                g->xy/n - (g->x/n)*(g->y/n), g->yy/n - (g->y/n)*(g->y/n), g->yz/n - (g->y/n)*(g->z/n),
                g->xz/n - (g->x/n)*(g->z/n), g->yz/n - (g->y/n)*(g->z/n), g->zz/n - (g->z/n)*(g->z/n);

        // Eigen decomposition for shape analysis
        Eigen::SelfAdjointEigenSolver<Mat3> es(g->cov);
        Point evals = es.eigenvalues();

        Scalar l0 = evals(0);
        Scalar l1 = evals(1);
        Scalar l2 = evals(2);

        Scalar linearity  = (l2 - l1) / l2;
        Scalar planarity  = (l1 - l0) / l2;
        Scalar scattering = l0 / l2;

        // Primitive classification
        if (planarity > linearity && planarity > scattering) {
            g->type = PrimitiveType::PLANE;

            // Plane normal: eigenvector of smallest eigenvalue
            Point evecMin = es.eigenvectors().col(0);
            g->n_vec = evecMin.normalized();
        }
        else if (linearity > planarity && linearity > scattering) {
            g->type = PrimitiveType::LINE;

            // Line direction: eigenvector of largest eigenvalue
            Point evecMax = es.eigenvectors().col(2);
            g->direction = evecMax.normalized();
        }
        else {
            g->type = PrimitiveType::VOLUME;

            // For volumes, nothing extra needed — mean and cov already store 3D Gaussian
        }

        node->init_node = true;
    }

    /**
     * @brief Re-fits plane and checks for neighbor merging (DSU Union)
     */
    void updateGeometry(UnionFindNode* node, Eigen::Vector3i& key) {
        initGeometry(node); // Update internal stats first

        // Neighbor Search for Merging (DSU)
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        int offsets[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,1}, {0,0,-1}};
        
        for (auto& off : offsets) {
            Eigen::Vector3i neighbor_key(key.x() + off[0], key.y() + off[1], key.z() + off[2]);
            auto it = map_.find(neighbor_key);
            if (it != map_.end()) {
                UnionFindNode* neighbor_root = it->second->find();
                if (neighbor_root != node) {
                    checkAndMerge(node, neighbor_root, v_size_);
                }
            }
        }
    }

    void checkAndMerge(UnionFindNode* root_a, UnionFindNode* root_b, Scalar &voxel_size) {
        GaussPtr ga = root_a->gauss_ptr;
        GaussPtr gb = root_b->gauss_ptr;

        // Only merge same type primitives
        if (ga->type != gb->type) return;

        bool can_merge = false;

        switch (ga->type) {
            case PrimitiveType::PLANE: {
                // --- Check normal alignment ---
                Scalar cos_theta = ga->n_vec.dot(gb->n_vec);
                cos_theta = std::clamp(cos_theta, Scalar(-1.0), Scalar(1.0));
                Scalar angle = std::acos(cos_theta); // radians
                constexpr Scalar max_angle = 5.0 * M_PI / 180.0; // 5 degrees
                if (angle > max_angle) break;

                // --- Check plane distance along normal ---
                Point diff = gb->mean - ga->mean;
                Scalar dist = std::abs(diff.dot(ga->n_vec));
                Scalar max_dist = voxel_size; // threshold can be tuned
                if (dist > max_dist) break;

                can_merge = true;
                break;
            }

            case PrimitiveType::LINE: {
                // --- Angle between line directions ---
                Scalar cos_angle = ga->direction.dot(gb->direction);
                cos_angle = std::clamp(cos_angle, Scalar(-1.0), Scalar(1.0));
                Scalar angle = std::acos(cos_angle);
                constexpr Scalar max_angle = 5.0 * M_PI / 180.0; // 5 degrees
                if (angle > max_angle) break;

                // --- Distance between lines ---
                Point delta = gb->mean - ga->mean;
                Point cross_dir = ga->direction.cross(delta);
                Scalar dist = cross_dir.norm();
                Scalar max_dist = voxel_size; // threshold can be tuned
                if (dist > max_dist) break;

                can_merge = true;
                break;
            }

            case PrimitiveType::VOLUME: {
                // --- Mahalanobis distance on means for Gaussian similarity ---
                Point diff = gb->mean - ga->mean;
                Mat3 cov_sum = ga->cov + gb->cov;
                cov_sum += Mat3::Identity() * 1e-6; // regularization
                if (cov_sum.determinant() < 1e-12) break;

                Scalar m_dist = std::sqrt(diff.transpose() * cov_sum.inverse() * diff);
                Scalar max_mdist = voxel_size; // threshold can be tuned
                if (m_dist > max_mdist) break;

                can_merge = true;
                break;
            }
        }

        if (!can_merge) return;

        // Union operation
        root_b->rootNode = root_a;

        // Merge Gaussian statistics (all types use same formula)
        int nA = ga->count;
        int nB = gb->count;
        int n  = nA + nB;

        // Merge means
        Point mean = (nA * ga->mean + nB * gb->mean) / n;

        // Merge covariances (scatter matrix update)
        Mat3 cov =
            (nA * (ga->cov + (ga->mean - mean) * (ga->mean - mean).transpose()) +
            nB * (gb->cov + (gb->mean - mean) * (gb->mean - mean).transpose())) / n;

        ga->mean  = mean;
        ga->cov   = cov;
        ga->count = n;

        // Merge additional type-specific vectors
        if (ga->type == PrimitiveType::PLANE) {
            ga->n_vec = ((nA * ga->n_vec + nB * gb->n_vec) / n).normalized();
        } else if (ga->type == PrimitiveType::LINE) {
            ga->direction = ((nA * ga->direction + nB * gb->direction) / n).normalized();
        }

        // Remove primitive from non-root
        root_b->gauss_ptr.reset(); 
    }

    mutable std::shared_mutex map_mtx_;
    std::unordered_map<Eigen::Vector3i, UnionFindNode*, HashVec3i> map_;
    Scalar v_size_, inv_v_size_;
    std::size_t update_threshold_;
};

} // namespace gauss_ivox_mapping