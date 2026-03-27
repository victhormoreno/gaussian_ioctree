#include <Eigen/Dense>
#include <unordered_map>
#include <vector>
#include <shared_mutex>
#include <memory>
#include <atomic>
#include <cmath>

#define HASH_P 116101
#define MAX_N 10000000000

namespace gauss_mapping {

using V3D = Eigen::Vector3d;
using M3D = Eigen::Matrix3d;

/** * @brief Adjugate matrix for 3x3 to avoid full inverse during covariance propagation
 */
inline void adjugateM3D(const M3D& A, M3D& A_star) {
    A_star(0, 0) = A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    A_star(0, 1) = A(0, 2) * A(2, 1) - A(0, 1) * A(2, 2);
    A_star(0, 2) = A(0, 1) * A(1, 2) - A(0, 2) * A(1, 1);
    A_star(1, 0) = A(1, 2) * A(2, 0) - A(1, 0) * A(2, 2);
    A_star(1, 1) = A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0);
    A_star(1, 2) = A(0, 2) * A(1, 0) - A(0, 0) * A(1, 2);
    A_star(2, 0) = A(1, 0) * A(2, 1) - A(1, 1) * A(2, 0);
    A_star(2, 1) = A(0, 1) * A(2, 0) - A(0, 0) * A(2, 1);
    A_star(2, 2) = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
}

class VOXEL_LOC
{
public:
    int64_t x, y, z;

    VOXEL_LOC(int64_t vx, int64_t vy, int64_t vz) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const
    {
        return (x == other.x && y == other.y && z == other.z);
    }
};

// Hash value
namespace std
{
    template <>
    struct hash<VOXEL_LOC>
    {
        int64_t operator()(const VOXEL_LOC &s) const
        {
            using std::hash;
            using std::size_t;
            return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
        }
    };
}

/*** 3D Point with Covariance ***/
typedef struct pointWithCov
{
    V3D point;
    V3D point_world;
    Eigen::Matrix3d cov;
    int Semantic_ID = -1;
} pointWithCov;

/*** Plane Structure ***/
typedef struct Plane
{
    /*** Update Flag ***/
    bool is_plane = false;
    bool is_init = false;

    /*** Plane Param ***/
    int main_direction = 0;
    M3D plane_cov;          
    V3D n_vec;

    /*** Incremental Calculation Param ***/
    double xx = 0.0;
    double yy = 0.0;
    double zz = 0.0;
    double xy = 0.0;
    double xz = 0.0;
    double yz = 0.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    V3D center = V3D::Zero();
    Eigen::Matrix3d covariance = M3D::Zero();
    int points_size = 0;
} Plane;
typedef std::shared_ptr<Plane> PlanePtr;
typedef const std::shared_ptr<Plane> PlaneConstPtr;

class UnionFindNode {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PlanePtr plane_ptr_;
    UnionFindNode *rootNode;
    
    // Instead of std::vector<pointWithCov>, we store points only until 
    // the first InitPlane, then use Incremental calculation.
    std::vector<pointWithCov> buffer_points; 
    
    bool init_node_ = false;
    bool is_plane = false;
    bool is_NDT = false;
    int semantic_category = -1;
    int num_points = 0;
    double voxel_center_[3];

    UnionFindNode(const VOXEL_LOC& pos, double voxel_sz) {
        plane_ptr_ = std::make_shared<Plane>();
        rootNode = this;
        voxel_center_[0] = (pos.x + 0.5) * voxel_sz;
        voxel_center_[1] = (pos.y + 0.5) * voxel_sz;
        voxel_center_[2] = (pos.z + 0.5) * voxel_sz;
    }

    // Path Compression for DSU
    UnionFindNode* find() {
        if (rootNode == this) return this;
        return rootNode = rootNode->find();
    }
};

class GaussianIVox {
public:
    GaussianIVox(double v_sz, int upd_thresh, double plane_thresh) 
        : voxel_size_(v_sz), update_threshold_(upd_thresh), planer_threshold_(plane_thresh) {}

    /**
     * @brief Updates the map with a new point. Handles DSU merging and geometry.
     */
    void update(const pointWithCov& pv) {
        VOXEL_LOC pos(
            static_cast<int64_t>(std::floor(pv.point[0] / voxel_size_)),
            static_cast<int64_t>(std::floor(pv.point[1] / voxel_size_)),
            static_cast<int64_t>(std::floor(pv.point[2] / voxel_size_))
        );

        UnionFindNode* node = nullptr;
        {
            std::unique_lock<std::shared_mutex> lock(map_mtx_);
            auto it = feat_map_.find(pos);
            if (it == feat_map_.end()) {
                node = new UnionFindNode(pos, voxel_size_);
                feat_map_[pos] = node;
            } else {
                node = it->second;
            }
        }

        // Always operate on the root for DSU consistency
        UnionFindNode* root = node->find();
        root->buffer_points.push_back(pv);
        root->num_points++;

        // Trigger geometry update if threshold reached
        if (root->buffer_points.size() >= update_threshold_) {
            if (!root->init_node_) {
                initPlaneGeometry(root);
            } else {
                updatePlaneGeometry(root, pos);
            }
            root->buffer_points.clear(); // Clear buffer to save memory
        }
    }

    /**
     * @brief Thread-safe retrieval of all unique planes in the map
     */
    std::vector<PlanePtr> getActivePlanes() const {
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        std::vector<PlanePtr> result;
        for (auto const& [loc, node] : feat_map_) {
            // Only return the root of a Union-Find set to avoid duplicates
            if (node->rootNode == node && node->is_plane) {
                result.push_back(node->plane_ptr_);
            }
        }
        return result;
    }

private:
    /**
     * @brief Initial Plane Fitting and Eigen Decomposition
     */
    void initPlaneGeometry(UnionFindNode* node) {
        auto& p = node->plane_ptr_;
        
        // 1. Incremental Summation (Updating existing Plane stats)
        for (const auto& pv : node->buffer_points) {
            p->points_size++;
            p->x += pv.point[0]; p->y += pv.point[1]; p->z += pv.point[2];
            p->xx += pv.point[0]*pv.point[0]; p->yy += pv.point[1]*pv.point[1]; p->zz += pv.point[2]*pv.point[2];
            p->xy += pv.point[0]*pv.point[1]; p->xz += pv.point[0]*pv.point[2]; p->yz += pv.point[1]*pv.point[2];
        }

        double n = static_cast<double>(p->points_size);
        p->center << p->x/n, p->y/n, p->z/n;

        // 2. Covariance Calculation
        p->covariance << p->xx/n - (p->x/n)*(p->x/n), p->xy/n - (p->x/n)*(p->y/n), p->xz/n - (p->x/n)*(p->z/n),
                         p->xy/n - (p->x/n)*(p->y/n), p->yy/n - (p->y/n)*(p->y/n), p->yz/n - (p->y/n)*(p->z/n),
                         p->xz/n - (p->x/n)*(p->z/n), p->yz/n - (p->y/n)*(p->z/n), p->zz/n - (p->z/n)*(p->z/n);

        // 3. Eigen Solver for SVD Plane Fitting
        Eigen::SelfAdjointEigenSolver<M3D> es(p->covariance);
        V3D evals = es.eigenvalues();
        
        if (evals(0) < planer_threshold_) {
            V3D evecMin = es.eigenvectors().col(0);
            solvePlaneParams(node, evecMin); // Logic for main_direction
            node->is_plane = true;
            node->init_node_ = true;
        }
    }

    /**
     * @brief Re-fits plane and checks for neighbor merging (DSU Union)
     */
    void updatePlaneGeometry(UnionFindNode* node, VOXEL_LOC& pos) {
        initPlaneGeometry(node); // Update internal stats first

        // Neighbor Search for Merging (DSU)
        std::shared_lock<std::shared_mutex> lock(map_mtx_);
        int offsets[6][3] = {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,1}, {0,0,-1}};
        
        for (auto& off : offsets) {
            VOXEL_LOC neighbor_pos(pos.x + off[0], pos.y + off[1], pos.z + off[2]);
            auto it = feat_map_.find(neighbor_pos);
            if (it != feat_map_.end()) {
                UnionFindNode* neighbor_root = it->second->find();
                if (neighbor_root != node && neighbor_root->is_plane) {
                    checkAndMerge(node, neighbor_root);
                }
            }
        }
    }

    void checkAndMerge(UnionFindNode* root_a, UnionFindNode* root_b) {
        PlanePtr pa = root_a->plane_ptr_;
        PlanePtr pb = root_b->plane_ptr_;
        
        if (pa->main_direction == pb->main_direction) {
            V3D diff = (pa->n_vec - pb->n_vec).cwiseAbs();
            // Mahalanobis distance check for plane similarity
            double m_dist = std::sqrt(diff.transpose() * (pa->plane_cov + pb->plane_cov).inverse() * diff);
            
            if (m_dist < 0.004 || (diff[0] < 0.1 && diff[1] < 0.1)) {
                root_b->rootNode = root_a; // Union operation
                // Simplified Covariance blending
                double weight_a = pa->points_size / static_cast<double>(pa->points_size + pb->points_size);
                double weight_b = 1.0 - weight_a;
                pa->n_vec = weight_a * pa->n_vec + weight_b * pb->n_vec;
            }
        }
    }

    /**
     * @brief Implementation of the 3-case projection (x+ay+bz+d=0, etc.)
     */
    void solvePlaneParams(UnionFindNode* node, const V3D& evecMin) {
        auto& p = node->plane_ptr_;
        M3D A, A_star;
        V3D E;
        double n = p->points_size;

        // Choose projection based on normal vector dominance
        if (std::abs(evecMin[0]) >= std::abs(evecMin[1]) && std::abs(evecMin[0]) >= std::abs(evecMin[2])) {
            p->main_direction = 2; // x-dominant
            E << -p->xy, -p->xz, -p->x;
            A << p->yy, p->yz, p->y, p->yz, p->zz, p->z, p->y, p->z, n;
        } else if (std::abs(evecMin[1]) >= std::abs(evecMin[0]) && std::abs(evecMin[1]) >= std::abs(evecMin[2])) {
            p->main_direction = 1; // y-dominant
            E << -p->xy, -p->yz, -p->y;
            A << p->xx, p->xz, p->x, p->xz, p->zz, p->z, p->x, p->z, n;
        } else {
            p->main_direction = 0; // z-dominant
            E << -p->xz, -p->yz, -p->z;
            A << p->xx, p->xy, p->x, p->xy, p->yy, p->y, p->x, p->y, n;
        }

        double det = A.determinant();
        if (std::abs(det) > 1e-9) {
            adjugateM3D(A, A_star);
            p->n_vec = A_star * E / det;
            // Note: Full plane_cov propagation requires looping points once or using 
            // the ddetA_dpw logic from your snippet if specific uncertainty is needed.
        }
    }

    mutable std::shared_mutex map_mtx_;
    std::unordered_map<VOXEL_LOC, UnionFindNode*> feat_map_;
    double voxel_size_;
    int update_threshold_;
    double planer_threshold_;
};

} // namespace gauss_mapping