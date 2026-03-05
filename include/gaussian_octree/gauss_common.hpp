#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <limits>
#include <deque>

namespace gauss_mapping {

using Scalar = double;
using Point  = Eigen::Matrix<Scalar, 3, 1>;
using Point4 = Eigen::Matrix<Scalar, 4, 1>;
using Mat3   = Eigen::Matrix<Scalar, 3, 3>;

struct PointCov {
    PointCov() { pos.setZero(); cov.setZero(); }
    PointCov(const Point& p): pos(p) { cov.setZero(); }
    PointCov(const Point& p, Scalar& c): pos(p) { cov = c * Mat3::Identity(); }
    PointCov(const Point& p, const Mat3& c): pos(p), cov(c) {}
    Point pos;
    Mat3 cov; 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using VecPointCov = std::vector<PointCov, Eigen::aligned_allocator<PointCov>>;

struct Gaussian {
    Point mean;
    Mat3 cov;
    Mat3 R_map; // AKF: Adaptive Measurement Noise Covariance
    int count = 0;

    Gaussian(): mean(Point::Zero()),
                cov(Mat3::Zero()),
                R_map(Mat3::Zero()),
                count(0) { }
    Gaussian(const PointCov& pt) { init(pt); }

    void init(const PointCov& pt)
    {
        mean = pt.pos;
        cov = pt.cov;
        R_map = pt.cov;
        count = 1;
    }

    bool checkFit(const Point& p_pos, const Mat3& p_cov, const Scalar& threshold) const {
        Point diff = p_pos - mean;

        // Total uncertainty = Geometric spread + Adaptive measurement noise + New point noise
        // In AKF-LIO, R_map already 'absorbs' the historical point noise.
        Mat3 S = getCovariance() + R_map + p_cov; 
        
        Scalar mahalanobis_dist = diff.transpose() * S.ldlt().solve(diff);

        return mahalanobis_dist < threshold;
    }
    
    // Adaptive Fusion from AKF-LIO paper
    void fuseAdaptive(const PointCov& pt, const Mat3& P_curr) {
        Point normal = getNormal();

        Scalar residual = normal.dot(pt.pos - mean);
        Scalar HPH = normal.transpose() * P_curr * normal;
        Scalar Rn = residual * residual + HPH;

        const Scalar R_MIN = 1e-6;
        const Scalar R_MAX = 1e-2;   
        Rn = std::clamp(Rn, R_MIN, R_MAX);

        // Mat3 R_obs = Mat3::Identity() * Rn;
        Mat3 R_obs = Rn * (normal * normal.transpose());

        Scalar alpha = static_cast<Scalar>(count) / (count + 1);
        R_map = alpha * R_map + (1.0 - alpha) * R_obs;

        addPoint(pt);
    }

    void merge(const Gaussian& other) {
        Scalar n1 = static_cast<Scalar>(this->count);
        Scalar n2 = static_cast<Scalar>(other.count);
        Scalar n_total = n1 + n2;

        // Save old mean for covariance calculation
        Point mu1 = this->mean;
        Point mu2 = other.mean;

        // Update Mean
        this->mean = (n1 * mu1 + n2 * mu2) / n_total;

        // Update Covariance using the Parallel Axis Theorem logic
        Mat3 term1 = n1 * (this->cov + mu1 * mu1.transpose());
        Mat3 term2 = n2 * (other.cov + mu2 * mu2.transpose());
        this->cov = (term1 + term2) / n_total - (this->mean * this->mean.transpose());
        
        // Update Adaptive Noise (AKF-LIO specific)
        this->R_map = (n1 * this->R_map + n2 * other.R_map) / n_total;

        this->count = static_cast<int>(n_total);
        // Re-symmetrize to prevent drift
        this->cov = 0.5 * (this->cov + this->cov.transpose().eval());
    }

    void addPoint(const PointCov& pt) {
        Point delta = pt.pos - mean;
        count++;

        mean += delta / Scalar(count);

        cov += delta * (pt.pos - mean).transpose();

        cov = 0.5 * (cov + cov.transpose().eval()); // keep symmetric
    }

    Mat3 getCovariance() const {
        return cov / Scalar(count); // geometric covariance
    }

    Point getNormal() const {
        // Get the normal (eigenvector of the smallest eigenvalue)
        Eigen::SelfAdjointEigenSolver<Mat3> es(getCovariance());
        Point normal = es.eigenvectors().col(0); 

        return normal;
    }

    Point4 getPlaneEquation() const {
        // Get the normal (eigenvector of the smallest eigenvalue)
        Eigen::SelfAdjointEigenSolver<Mat3> es(getCovariance());
        Point normal = es.eigenvectors().col(0); 

        // Calculate D (distance from origin to plane)
        // D = -(Ax + By + Cz)
        Scalar D = -normal.dot(mean);

        return Point4(normal.x(), normal.y(), normal.z(), D);
    }

    Scalar getPlaneDistance(const Point& p) {
        Point4 n_ABCD = getPlaneEquation();
        return n_ABCD(0) * p(0) + n_ABCD(1) * p(1) + n_ABCD(2) * p(2) + n_ABCD(3);
    }

    Scalar getMahalanobisDistance(const Point& p) {
        Point diff = p - mean;

        return diff.transpose() * getCovariance().ldlt().solve(diff);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct Octant {

    Point centroid;
    Scalar extent;

    std::vector<Gaussian*> gaussians;

    Octant **children;

    Octant() : centroid(Point::Zero()), extent(0.0), children(nullptr) { }

    ~Octant() {
        if (!isLeaf()) {
            for (int i = 0; i < 8; ++i) {
                if (children[i] != nullptr)
                delete children[i];
            }

            delete[] children;
            children = nullptr;
        }
        gaussians.clear();
    }

    bool isLeaf() const
    {
        return children == nullptr;
    }

    void init_children()
    {
        children = new Octant*[8]();
    }
};

struct Neighbor {
    Scalar dist_sq;
    Gaussian g;
    // Max-heap comparison
    bool operator<(const Neighbor& other) const { return dist_sq < other.dist_sq; }
};

class NeighborHeap {
public:
    explicit NeighborHeap(size_t k) : k_(k) { data_.reserve(k); }

    void add(const Gaussian& g, Scalar d2) {
        if (data_.size() < k_) {
            data_.push_back({d2, g});
            std::push_heap(data_.begin(), data_.end());
        } else if (d2 < data_.front().dist_sq) {
            std::pop_heap(data_.begin(), data_.end());
            data_.back() = {d2, g};
            std::push_heap(data_.begin(), data_.end());
        }
    }

    bool full() const { return data_.size() >= k_; }
    Scalar worstDist() const { return data_.empty() ? std::numeric_limits<Scalar>::max() : data_.front().dist_sq; }
    const std::vector<Neighbor>& data() const { return data_; }
    bool empty() const {return data_.empty(); }

private:
    size_t k_;
    std::vector<Neighbor> data_;
};

template <typename T>
class Pool {
public:
    Pool() = default;

    // Allocate a new object from the pool
    T* allocate() 
    {
        if (!free_list_.empty()) 
        {
            T* obj = free_list_.back();
            free_list_.pop_back();

            *obj = T(); // reset contents
            return obj;
        }

        storage_.emplace_back();
        return &storage_.back();
    }

    // Free an object back into the pool
    void free(T* obj) {
        free_list_.push_back(obj);
    }

    // Clear entire pool
    void clear() {
        storage_.clear();
        free_list_.clear();
    }

    // Number of allocated objects (excluding free ones)
    size_t size() const {
        return storage_.size() - free_list_.size();
    }

    // Retrieve active objects
    std::vector<T*> getActive()
    {
        std::vector<T*> result;
        result.reserve(size());

        for (auto& obj : storage_)
        {
            T* ptr = &obj;

            if (std::find(free_list_.begin(), free_list_.end(), ptr) == free_list_.end())
                result.push_back(ptr);
        }

        return result;
    }

private:
    std::deque<T> storage_;        // stable memory (no realloc)
    std::vector<T*> free_list_;    // reusable objects
};

} // namespace gauss_mapping