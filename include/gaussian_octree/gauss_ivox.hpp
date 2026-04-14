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

inline void adjugateM3D(const Mat3& A, Mat3& adj) {
    adj(0,0) =  A(1,1)*A(2,2) - A(1,2)*A(2,1);
    adj(0,1) = -A(0,1)*A(2,2) + A(0,2)*A(2,1);
    adj(0,2) =  A(0,1)*A(1,2) - A(0,2)*A(1,1);

    adj(1,0) = -A(1,0)*A(2,2) + A(1,2)*A(2,0);
    adj(1,1) =  A(0,0)*A(2,2) - A(0,2)*A(2,0);
    adj(1,2) = -A(0,0)*A(1,2) + A(0,2)*A(1,0);

    adj(2,0) =  A(1,0)*A(2,1) - A(1,1)*A(2,0);
    adj(2,1) = -A(0,0)*A(2,1) + A(0,1)*A(2,0);
    adj(2,2) =  A(0,0)*A(1,1) - A(0,1)*A(1,0);
}

/*** 3D Point with Covariance ***/
struct pointWithCov
{
    Point point;
    Mat3 cov;
};

enum class PrimitiveType {
    UNKNOWN,
    LINE, // --- To-Do IGNORE ---
    PLANE,
    VOLUME
};

struct GaussianPrimitive
{
    // --- Core Gaussian ---
    Point mean = Point::Zero(); // world coordinates mean
    Mat3  cov  = Mat3::Zero();
    PrimitiveType type = PrimitiveType::UNKNOWN;

    // --- Plane parametric form ax + by + cz + d = 0 ---
    int main_dir; // 0=z dominant, 1=y dominant, 2=x dominant
    Point n_vec;       // Plane normal vector
    Scalar d;          // Plane offset
    Point param;       // [a, b, d] plane coefficients in dominant axis form
    Mat3 plane_cov = Mat3::Zero();

    // --- Incremental stats (Scatter Matrix) ---
    Scalar xx = 0, yy = 0, zz = 0;
    Scalar xy = 0, xz = 0, yz = 0;
    Scalar x = 0, y = 0, z = 0;
    int count = 0;

    // Updates sums immediately upon point arrival
    void addPoint(const Point& p) {
        count++;
        x += p.x(); y += p.y(); z += p.z();
        xx += p.x() * p.x(); yy += p.y() * p.y(); zz += p.z() * p.z();
        xy += p.x() * p.y(); xz += p.x() * p.z(); yz += p.y() * p.z();
    }

    // Merges sums from another primitive during DSU Union
    void mergeSums(const GaussianPrimitive& other, const Point& delta) {
        Scalar n2 = static_cast<Scalar>(other.count);

        // Shift other's linear sums: x' = x + n*delta
        Scalar nx_new = other.x + n2 * delta.x();
        Scalar ny_new = other.y + n2 * delta.y();
        Scalar nz_new = other.z + n2 * delta.z();

        // Shift other's square/cross sums (Parallel Axis Theorem logic)
        // xx' = sum((xi + dx)^2) = sum(xi^2 + 2*xi*dx + dx^2) = xx + 2*dx*x + n*dx^2
        xx += (other.xx + 2.0 * delta.x() * other.x + n2 * delta.x() * delta.x());
        yy += (other.yy + 2.0 * delta.y() * other.y + n2 * delta.y() * delta.y());
        zz += (other.zz + 2.0 * delta.z() * other.z + n2 * delta.z() * delta.z());
        
        xy += (other.xy + delta.x() * other.y + delta.y() * other.x + n2 * delta.x() * delta.y());
        xz += (other.xz + delta.x() * other.z + delta.z() * other.x + n2 * delta.x() * delta.z());
        yz += (other.yz + delta.y() * other.z + delta.z() * other.y + n2 * delta.y() * delta.z());

        x += nx_new;
        y += ny_new;
        z += nz_new;
        count += other.count;

        mean = (mean * (count - other.count) + other.mean * other.count) / count; // Update mean for merged primitive
    }
};

using GaussPtr = std::shared_ptr<GaussianPrimitive>;
using GaussConstPtr = const std::shared_ptr<GaussianPrimitive>;

class UnionFindNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GaussPtr gauss_ptr;
    UnionFindNode *rootNode;
    Point voxel_center; // Center of the voxel for relative point storage
    std::atomic<size_t> points_since_update = 0;
    
    UnionFindNode(const Point& center) : voxel_center(center) {
        gauss_ptr = std::make_shared<GaussianPrimitive>();
        rootNode = this;
    }

    // Path Compression for DSU
    UnionFindNode* find() {
        if (rootNode == this) return this;
        return rootNode = rootNode->find();
    }
};

class GaussianIVox {
public:

    GaussianIVox(Scalar v_sz, 
                std::size_t upd_thresh, 
                Scalar planarity_thresh = 0.01, 
                Scalar chi_square_thresh = 7.815,
                Scalar sensor_noise = 0.01) 
        : v_size_(v_sz), inv_v_size_(1.0 / v_sz), update_threshold_(upd_thresh),
          planarity_threshold_(planarity_thresh), chi_square_threshold_(chi_square_thresh), 
          noise_(sensor_noise) { }

    ~GaussianIVox() {
        for (auto& pair : map_) {
            delete pair.second;
        }
        map_.clear();
    }

    /**
     * @brief Updates the map with a new point. Handles DSU merging and geometry.
     */
    void update(const pointWithCov& pv) {
        Eigen::Vector3i key = (pv.point * inv_v_size_).array().floor().cast<int>();

        Point voxel_center;
        voxel_center[0] = (key.x() + 0.5) * v_size_;
        voxel_center[1] = (key.y() + 0.5) * v_size_;
        voxel_center[2] = (key.z() + 0.5) * v_size_;
        
        UnionFindNode* node = nullptr;
        {
            std::unique_lock<std::shared_mutex> lock(map_mtx_);
            auto it = map_.find(key);
            if (it == map_.end()) {
                node = new UnionFindNode(voxel_center);
                map_[key] = node;
            } else {
                node = it->second;
            }
        }

        UnionFindNode* root = node->find();
        root->gauss_ptr->addPoint(pv.point - voxel_center); // Store points relative to voxel center for better numerical stability
        root->points_since_update++;

        // Trigger geometry update incrementally
        if (root->points_since_update >= update_threshold_) {
            computeGeometry(root);
            checkNeighbors(root, key);
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

    /**
     * @brief Gaussian Fitting and Eigen Decomposition
     */
    void computeGeometry(UnionFindNode* node) {

        auto& g = node->gauss_ptr;
        if (g->count < 5) return;

        Scalar n = static_cast<Scalar>(g->count);

        // Recovery: World Mean = Local Mean + Voxel Center
        g->mean << (g->x / n) + node->voxel_center.x(),
                    (g->y / n) + node->voxel_center.y(),
                    (g->z / n) + node->voxel_center.z();

        // Unbiased covariance estimation using the scatter matrix and mean
        Scalar inv = 1.0 / (n - 1); // Sample covariance normalization

        Scalar mx = g->x / n;
        Scalar my = g->y / n;
        Scalar mz = g->z / n;

        g->cov <<
            (g->xx - n*mx*mx) * inv, (g->xy - n*mx*my) * inv, (g->xz - n*mx*mz) * inv,
            (g->xy - n*mx*my) * inv, (g->yy - n*my*my) * inv, (g->yz - n*my*mz) * inv,
            (g->xz - n*mx*mz) * inv, (g->yz - n*my*mz) * inv, (g->zz - n*mz*mz) * inv;

        Eigen::SelfAdjointEigenSolver<Mat3> es(g->cov);
        Point evals = es.eigenvalues();
        
        // PCA classification
        Scalar l0 = std::max(evals(0), 0.0);

        // Primitive classification
        // if (l0 < planarity_threshold_) {
        if (true) { // --- For now, keep all as planes for debugging purposes ---
            g->type = PrimitiveType::PLANE;

            if(!solvePlaneAdjugate(node, evals, es.eigenvectors().col(0)))
                g->type = PrimitiveType::UNKNOWN; // Degenerate case, keep as UNKNOWN
        }
        else {
            g->type = PrimitiveType::VOLUME;

            // For volumes, nothing extra needed — mean and cov already store 3D Gaussian
        }

        node->points_since_update = 0;
    }

    void checkNeighbors(UnionFindNode* root, const Eigen::Vector3i& key) {
        
        // 6-Neighbor Search for Merging (DSU)
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        int offsets[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,1}, {0,0,-1}};

        for (auto& off : offsets) {
            Eigen::Vector3i neighbor_key(key.x() + off[0], key.y() + off[1], key.z() + off[2]);
            auto it = map_.find(neighbor_key);
            if (it != map_.end()) {
                UnionFindNode* neighbor_root = it->second->find();
                if (neighbor_root != root) tryMerge(root, neighbor_root);
            }
        }
    }

    void tryMerge(UnionFindNode* a, UnionFindNode* b) {

        GaussPtr ga = a->gauss_ptr;
        GaussPtr gb = b->gauss_ptr;

        // Only merge same type primitives
        if (ga->type != gb->type || ga->type == PrimitiveType::UNKNOWN) return;

        bool merge = false;
        switch (ga->type) {
            case PrimitiveType::PLANE: {
                // --- Mahalanobis distance for Plane similarity ---
                if(ga->main_dir != gb->main_dir) break;

                if (ga->n_vec.dot(gb->n_vec) < 0) {
                    gb->param *= -1.0;
                }
                
                // (Bayesian style)
                Eigen::Vector3d delta_pi = ga->param - gb->param; 

                // Add a regularization term to the sum of covariances
                // This represents the uncertainty of the "link" between two voxels.
                Mat3 cov_sum = ga->plane_cov + gb->plane_cov;
                cov_sum.diagonal().array() += noise_*noise_; // regularization floor

                Scalar m_dist_sq = delta_pi.transpose() * cov_sum.inverse() * delta_pi;

                if (m_dist_sq < chi_square_threshold_)
                    merge = true;

                break;
            }

            case PrimitiveType::VOLUME: {
                // --- Mahalanobis distance on means for Gaussian similarity ---
                Point diff = gb->mean - ga->mean;
                Mat3 cov_sum = ga->cov + gb->cov;
                cov_sum.diagonal().array() += noise_*noise_; // regularization floor
                if (cov_sum.determinant() < 1e-12) break;

                Scalar m_dist_sq = diff.transpose() * cov_sum.inverse() * diff;

                if (m_dist_sq < chi_square_threshold_) 
                    merge = true;

                break;
            }

            default:
                break;
        }

        if(!merge) return;

        b->rootNode = a; // DSU Union: root of b points to root of a

        Point delta = b->voxel_center - a->voxel_center; // spatial shift
        ga->mergeSums(*(gb), delta); // Merge incremental sums for accurate geometry update

        switch (ga->type) {
            case PrimitiveType::PLANE: {
                // VoxelMap++ merging logic: weighted average of plane parameters based on covariance "confidence"
                Scalar trA = ga->plane_cov.trace();
                Scalar trB = gb->plane_cov.trace();

                Scalar wA = trB / (trA + trB);
                Scalar wB = trA / (trA + trB);
                
                ga->param = wA * ga->param + wB * gb->param;
                ga->plane_cov = wA * wA * ga->plane_cov + wB * wB * gb->plane_cov;

                // Re-build plane normal from merged parameters
                switch(ga->main_dir) {
                    case 0: ga->n_vec << ga->param[0], ga->param[1], 1.0; break; // z dominant
                    case 1: ga->n_vec << ga->param[0], 1.0, ga->param[1]; break; // y dominant
                    case 2: ga->n_vec << 1.0, ga->param[0], ga->param[1]; break; // x dominant
                    default: break; // should never happen
                }
                ga->n_vec.normalize();
                ga->d = -ga->n_vec.dot(ga->mean); // world-frame plane offset
                break;
            }
            
            case PrimitiveType::VOLUME: {
                computeGeometry(a); // Recompute mean and covariance for merged volume
                break;
            }

            default:
                break;
        }

        b->gauss_ptr.reset(); // Clear merged primitive to save memory, only root holds valid Gaussian
    }

    /**
    * @brief Solve plane using adjugate method
    *
    * @param g                       : Gaussian pointer
    * @param evecMin                 : smallest eigenvector (for stability choice)
    * @return false if degenerate
    */
    bool solvePlaneAdjugate(
        UnionFindNode* node,
        const Point& evals,
        const Point& evecMin
    ) {
        Mat3 A, A_star;
        Point E;
        Scalar detA = 0.0;

        auto g = node->gauss_ptr;

        Scalar n = static_cast<Scalar>(g->count);
        Scalar xx = g->xx, yy = g->yy, zz = g->zz;
        Scalar xy = g->xy, xz = g->xz, yz = g->yz;
        Scalar x  = g->x,  y  = g->y,  z  = g->z;

        // --- Select most stable parameterization ---
        g->main_dir = 0;

        if (std::fabs(evecMin[0]) >= std::fabs(evecMin[1]) &&
            std::fabs(evecMin[0]) >= std::fabs(evecMin[2])) {
            g->main_dir = 2; // x dominant → x + ay + bz + d = 0
            A << yy, yz, y,
                yz, zz, z,
                y,  z,  n;

            E << -xy, -xz, -x;
        }
        else if (std::fabs(evecMin[1]) >= std::fabs(evecMin[0]) &&
                std::fabs(evecMin[1]) >= std::fabs(evecMin[2])) {
            g->main_dir = 1; // y dominant → ax + y + bz + d = 0

            A << xx, xz, x,
                xz, zz, z,
                x,  z,  n;

            E << -xy, -yz, -y;
        }
        else {
            g->main_dir = 0; // z dominant → ax + by + z + d = 0

            A << xx, xy, x,
                xy, yy, y,
                x,  y,  n;

            E << -xz, -yz, -z;
        }

        detA = A.determinant();
        if (std::fabs(detA) < 1e-9) return false;

        adjugateM3D(A, A_star);

        Point param = (A_star * E) / detA;

        // --- Convert to parameterization (a,b,c) + offset (d) ---
        switch(g->main_dir) {
            case 0: g->n_vec << param[0], param[1], 1.0; break; // z dominant
            case 1: g->n_vec << param[0], 1.0, param[1]; break; // y dominant
            case 2: g->n_vec << 1.0, param[0], param[1]; break; // x dominant
            default: return false; // should never happen
        }

        // Normalize plane
        g->n_vec.normalize();

        // Save plane offset 
        g->d = -g->n_vec.dot(g->mean); // world-frame plane offset

        // --- Compute plane covariance ---
        // Calculate the variance of the residuals (sigma^2)
        // l0 is the mean squared error; we multiply by n to get the sum of squares
        Scalar residual_var = (evals(0) * n) / std::max(n - 3.0, 1.0);

        // Add Sensor Noise Floor (Hardware Limit)
        // VoxelMap++ suggests that the hardware noise is a constant offset 
        // to the residual variance.
        constexpr Scalar kSensorNoise = 0.0004; // (0.02m)^2
        Scalar total_sigma2 = residual_var + kSensorNoise;

        // The covariance of the solved parameters [a, b, d] is sigma^2 * A^-1
        // We already have A_star (adjugate) and detA.
        Mat3 C_param = (A_star / detA) * total_sigma2;

        Mat3 T = Mat3::Identity();
        switch(g->main_dir) {
            case 0: 
                T(2,0) = -node->voxel_center.x(); T(2,1) = -node->voxel_center.y(); 
                g->param[0] = g->n_vec[0]; g->param[1] = g->n_vec[1]; g->param[2] = g->d; // Store plane parameters for merging logic
                break;
            case 1: 
                T(2,0) = -node->voxel_center.x(); T(2,1) = -node->voxel_center.z(); 
                g->param[0] = g->n_vec[0]; g->param[1] = g->n_vec[2]; g->param[2] = g->d;
                break;
            case 2: 
                T(2,0) = -node->voxel_center.y(); T(2,1) = -node->voxel_center.z(); 
                g->param[0] = g->n_vec[1]; g->param[1] = g->n_vec[2]; g->param[2] = g->d;
                break;
            default:
                return false; // should never happen
        }

        // This is the covariance of [a, b, d_world]
        g->plane_cov = T * C_param * T.transpose();

        return true;
    }

    mutable std::shared_mutex map_mtx_;
    std::unordered_map<Eigen::Vector3i, UnionFindNode*, HashVec3i> map_;
    Scalar v_size_, inv_v_size_;
    std::size_t update_threshold_;
    Scalar planarity_threshold_;  // Tunable threshold for plane classification
    Scalar chi_square_threshold_; // Tunable threshold for plane merging
    Scalar noise_;                // Sensor noise floor for covariance estimation
};

} // namespace gauss_ivox_mapping