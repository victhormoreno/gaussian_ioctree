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

/**
 * @brief Combines two hash values using FNV-like mixing.
 * @param seed Initial hash seed
 * @param v Value to combine
 */
inline size_t hash_combine(size_t seed, size_t v) {
    return seed ^ (v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

struct HashVec3i {
    /**
     * @brief Hash function for 3D integer vectors
     * @param v 3D vector to hash
     */
    size_t operator()(const Eigen::Vector3i& v) const {
        size_t seed = 0;
        seed = hash_combine(seed, std::hash<int>()(v[0]));
        seed = hash_combine(seed, std::hash<int>()(v[1]));
        seed = hash_combine(seed, std::hash<int>()(v[2]));
        return seed;
    }
};

/**
 * @brief Computes the adjugate (adjoint) matrix of a 3x3 matrix
 * @param A Input 3x3 matrix
 * @param adj Output adjugate matrix
 */
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
    Point param;  // [a, b, d] plane parametrization in dominant axis form
    Mat3 plane_cov = Mat3::Zero();

    // --- Incremental stats (Scatter Matrix) ---
    Scalar xx = 0, yy = 0, zz = 0;
    Scalar xy = 0, xz = 0, yz = 0;
    Scalar x = 0, y = 0, z = 0;
    int count = 0;

    /**
     * @brief Updates incremental sums with a new point
     * @param p Point to add to the Gaussian primitive
     */
    void addPoint(const Point& p) {
        count++;
        x += p.x(); y += p.y(); z += p.z();
        xx += p.x() * p.x(); yy += p.y() * p.y(); zz += p.z() * p.z();
        xy += p.x() * p.y(); xz += p.x() * p.z(); yz += p.y() * p.z();
    }

    /**
     * @brief Merges incremental sums from another primitive using parallel axis theorem
     * @param other Another Gaussian primitive to merge
     * @param delta Spatial offset between voxel centers
     */
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
    }
};

using GaussPtr = std::shared_ptr<GaussianPrimitive>;

class UnionFindNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GaussPtr gauss_ptr;
    UnionFindNode *rootNode;
    Point voxel_center; // Center of the voxel for relative point storage
    std::atomic<size_t> points_since_update = 0;
    
    /**
     * @brief Constructs a union-find node for a voxel
     * @param center Center coordinates of the voxel
     */
    UnionFindNode(const Point& center) : voxel_center(center) {
        gauss_ptr = std::make_shared<GaussianPrimitive>();
        rootNode = this;
    }

    /**
     * @brief Finds root node with path compression
     */
    UnionFindNode* find() {
        if (rootNode == this) return this;
        return rootNode = rootNode->find();
    }
};

class GaussianIVox {
public:

    /**
     * @brief Constructs a GaussianIVox map with specified parameters
     * @param v_sz Voxel size (resolution)
     * @param upd_thresh Update threshold (points to accumulate before geometry recomputation)
     * @param planarity_thresh Threshold for plane vs volume classification
     * @param chi_square_thresh Mahalanobis distance threshold for merging
     * @param sensor_noise Sensor noise floor for covariance regularization
     */
    GaussianIVox(Scalar v_sz, 
                std::size_t upd_thresh, 
                Scalar planarity_thresh = 0.01, 
                Scalar chi_square_thresh = 7.815,
                Scalar sensor_noise = 0.01) 
        : v_size_(v_sz), inv_v_size_(1.0 / v_sz), update_threshold_(upd_thresh),
          planarity_threshold_(planarity_thresh), chi_square_threshold_(chi_square_thresh), 
          noise_(sensor_noise) { }

    /**
     * @brief Destructor: cleans up all union-find nodes
     */
    ~GaussianIVox() {
        for (auto& pair : map_) {
            delete pair.second;
        }
        map_.clear();
    }

    /**
     * @brief Updates the map with a new point, triggers geometry computation and neighbor merging
     * @param point Point to insert into the map (world-frame coordinates)
     */
    void update(const Point& point) {
        Eigen::Vector3i key = (point * inv_v_size_).array().floor().cast<int>();

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
        root->gauss_ptr->addPoint(point - voxel_center); // Store points relative to voxel center for better numerical stability
        root->points_since_update++;

        // Trigger geometry update incrementally
        if (root->points_since_update >= update_threshold_) {
            computeGeometry(root);
            checkNeighbors(root, key);
        }
    }

    /**
     * @brief Retrieves the Gaussian primitive at a given point location
     * @param p Query point in world coordinates
     * @return Shared pointer to the root Gaussian primitive, or nullptr if not found
     */
    GaussPtr getPrimitiveAtPoint(const Point& p) const
    {
        Eigen::Vector3i key = (p * inv_v_size_).array().floor().cast<int>();

        std::shared_lock<std::shared_mutex> lock(map_mtx_);

        auto it = map_.find(key);
        if (it == map_.end())
            return nullptr;

        UnionFindNode* node = it->second;
        UnionFindNode* root = node->find();

        return root->gauss_ptr;
    }

    /**
     * @brief Thread-safe retrieval of all unique root Gaussian primitives
     * @return Vector of shared pointers to all root Gaussians in the map
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
     * @brief Thread-safe retrieval of Gaussians filtered by primitive type
     * @param type Primitive type to filter (PLANE, VOLUME, etc.)
     * @return Vector of shared pointers to Gaussians of the specified type
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
     * @brief Constructs the plane normal vector based on dominant axis
     * @param p Plane parameters [a, b, d] in dominant axis form
     * @param main_dir Dominant axis (0=z, 1=y, 2=x)
     * @return Normal vector in form [x, y, z] == [a, b, c]
     */
    Point buildNormalVector(const Point& p, int main_dir) {
        Point w;
        switch (main_dir) {
            case 0: w << p[0], p[1], 1.0; break;
            case 1: w << p[0], 1.0, p[1]; break;
            case 2: w << 1.0, p[0], p[1]; break;
        }
        return w;
    }

    /**
     * @brief Computes Gaussian geometry from scatter matrix, classifies primitive type via PCA
     * @param node Union-find node containing the point statistics
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

    /**
     * @brief Checks 6-neighbor voxels and attempts merging with same-type primitives
     * @param root Root union-find node of current voxel
     * @param key Voxel grid key of current location
     */
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

    /**
     * @brief Attempts to merge two primitives using Mahalanobis distance test
     * @param a Root node of first primitive
     * @param b Root node of second primitive
     */
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

                Point na = buildNormalVector(ga->param, ga->main_dir);
                Point nb = buildNormalVector(gb->param, gb->main_dir);

                if (na.dot(nb) < 0) {
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
     * @brief Solves plane parameters using adjugate method, selects stable axis form
     * @param node Union-find node containing point statistics
     * @param evals Eigenvalues of covariance matrix
     * @param evecMin Smallest eigenvector for axis selection
     * @return true if plane solved successfully, false if degenerate
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

        // --- Transform plane offset to world-frame ---
        Point n_vec = buildNormalVector(param, g->main_dir);
        param[2] -= n_vec.dot(node->voxel_center);

        // Store dominant axis parameters [a, b, d_world] for merging logic
        g->param = param; 

        // --- Compute plane covariance ---
        // Calculate the variance of the residuals (sigma^2)
        // l0 is the mean squared error; we multiply by n to get the sum of squares
        Scalar residual_var = (evals(0) * n) / std::max(n - 3.0, 1.0);

        // Add Sensor Noise Floor (Hardware Limit)
        // VoxelMap++ suggests that the hardware noise is a constant offset 
        // to the residual variance.
        const Scalar kSensorNoise = noise_*noise_; 
        Scalar total_sigma2 = residual_var + kSensorNoise;

        // The covariance of the solved parameters [a, b, d] is sigma^2 * A^-1
        // We already have A_star (adjugate) and detA.
        Mat3 C_param = (A_star / detA) * total_sigma2;

        Mat3 T = Mat3::Identity();
        switch(g->main_dir) {
            case 0: 
                T(2,0) = -node->voxel_center.x(); T(2,1) = -node->voxel_center.y(); 
                break;
            case 1: 
                T(2,0) = -node->voxel_center.x(); T(2,1) = -node->voxel_center.z(); 
                break;
            case 2: 
                T(2,0) = -node->voxel_center.y(); T(2,1) = -node->voxel_center.z(); 
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