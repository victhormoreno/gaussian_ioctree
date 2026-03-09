#pragma once

#include "gaussian_octree/gauss_common.hpp"
#include "gaussian_octree/debug_timer.hpp"

namespace gauss_mapping {

class Octree {

public:

    Octree(size_t max_g = 5,
           Scalar min_e = 0.1,
           Scalar chi = 11.345 )
        : max_gaussians_leaf_(max_g),
          min_extent_(min_e),
          chi_threshold_(chi)
    { }

private:

    /*
    ========================================
    MEMORY POOLS
    ========================================
    */

    Pool<Octant> octant_pool_;
    Pool<Gaussian> gaussian_pool_;

    Octant* allocOctant()
    {
        return octant_pool_.allocate();
    }

    Gaussian* allocGaussian(const PointCov& pt)
    {
        Gaussian* g = gaussian_pool_.allocate();
        g->init(pt);
        return g;
    }

    /*
    ========================================
    TREE STATE
    ========================================
    */

    Octant* root_ = nullptr;

    size_t num_points_ = 0;
    size_t max_gaussians_leaf_;

    Scalar min_extent_;
    Scalar chi_threshold_;

public:

    void clear()
    {
        if(root_)
            freeSubtree(root_);

        root_ = nullptr;

        octant_pool_.clear();
        gaussian_pool_.clear();

        num_points_ = 0;
    }

    size_t size() const { return gaussian_pool_.size(); }
    size_t num_points() const { return num_points_; }
    size_t num_octants() const { return octant_pool_.size(); }

    /*
    ========================================
    UPDATE ENTRY
    ========================================
    */

    void update(const VecPointCov& pts, const Mat3& P_curr)
    {
        DEBUG_PROFILE_SCOPE("Octree::update");

        if(pts.empty()) return;

        Point min = Point::Constant(std::numeric_limits<Scalar>::max());
        Point max = Point::Constant(std::numeric_limits<Scalar>::lowest());

        for(const auto& p:pts)
        {
            min = min.cwiseMin(p.pos);
            max = max.cwiseMax(p.pos);
        }

        if(!root_)
        {
            initialize(pts, min, max);
            return;
        }

        expandTree(max);
        expandTree(min);

        updateOctant(root_, pts, P_curr);
    }

    // k-Nearest Neighbors
    void knnSearch(const Point& query, int k,
                   std::vector<Gaussian*>& neighbors,
                   std::vector<Scalar>& distances) const
    {
        DEBUG_PROFILE_SCOPE("Octree::knnSearch");

        if(!root_) return;

        NeighborHeap heap(k);
        knnRecursive(root_, query, heap);

        neighbors.clear();
        distances.clear();
        for(const auto& n : heap.data())
        {
            neighbors.push_back(const_cast<Gaussian*>(&n.g)); // return pointer
            distances.push_back(std::sqrt(n.dist_sq));
        }
    }

    // Radius search
    void radiusSearch(const Point& query,
                      Scalar radius,
                      std::vector<Gaussian*>& neighbors,
                      std::vector<Scalar>& distances) const
    {
        DEBUG_PROFILE_SCOPE("Octree::radiusSearch");

        if(!root_) return;

        neighbors.clear();
        distances.clear();

        radiusRecursive(root_, query, radius*radius, neighbors, distances);
    }

    std::vector<Gaussian*> getGaussians()
    {
        return gaussian_pool_.getActive();
    }

    std::vector<Octant*> getOctants()
    {
        return octant_pool_.getActive();
    }

    template <typename PointT, typename ContainerT>
    ContainerT getData() {
        auto gauss = getGaussians();

        ContainerT out;
        out.reserve(gauss.size());

        for (auto* g : gauss) {
            PointT pt;
            pt.x = g->mean.x();
            pt.y = g->mean.y();
            pt.z = g->mean.z();
            pt.intensity = g->R_map.trace();
            out.push_back(pt);
        }

        return out;
    }

private:

    /*
    ========================================
    INITIALIZATION
    ========================================
    */

    void initialize(const VecPointCov& pts, const Point& min, const Point& max)
    {
        DEBUG_PROFILE_SCOPE("Octree::initialize");

        clear();

        Point extent = 0.5 * (max - min);
        Point center = min + extent;

        root_ = createOctant(center, extent.maxCoeff(), pts);
    }

    /*
    ========================================
    OCTANT CREATION
    ========================================
    */

    Octant* createOctant(const Point& centroid,
                         Scalar extent,
                         const VecPointCov& points)
    {
        DEBUG_PROFILE_SCOPE("Octree::createOctant");

        Octant* oct = allocOctant();

        oct->centroid = centroid;
        oct->extent = extent;

        if(points.size() > max_gaussians_leaf_
           && extent > 2*min_extent_)
        {
            oct->init_children();

            std::vector<VecPointCov> child_pts(8);

            for(const auto& p:points)
                child_pts[getIdx(p.pos,centroid)].push_back(p);

            for(int i=0;i<8;i++)
            {
                if(child_pts[i].empty()) continue;

                Point c = computeChildCenter(centroid, extent, i);

                oct->children[i] =
                    createOctant(c, 0.5*extent, child_pts[i]);
            }
        }
        else
        {
            for(const auto& p:points)
            {
                oct->gaussians.push_back(allocGaussian(p));
                num_points_++;
            }
        }

        return oct;
    }

    /*
    ========================================
    ROOT EXPANSION
    ========================================
    */

    void expandTree(const Point& boundary)
    {
        DEBUG_PROFILE_SCOPE("Octree::expandTree");

        static const Scalar factor[] = {-0.5,0.5};

        while((boundary-root_->centroid).cwiseAbs().maxCoeff()
              > root_->extent)
        {
            Scalar parent_extent = 2*root_->extent;

            Point parent_centroid(
                root_->centroid.x()+factor[boundary.x()>root_->centroid.x()]*parent_extent,
                root_->centroid.y()+factor[boundary.y()>root_->centroid.y()]*parent_extent,
                root_->centroid.z()+factor[boundary.z()>root_->centroid.z()]*parent_extent
            );

            Octant* parent = allocOctant();

            parent->centroid = parent_centroid;
            parent->extent = parent_extent;
            parent->init_children();

            parent->children[getIdx(root_->centroid, parent_centroid)] = root_;

            root_ = parent;
        }
    }

    /*
    ========================================
    OCTANT UPDATE
    ========================================
    */

    void updateOctant(Octant*& oct,
                      const VecPointCov& points,
                      const Mat3& P_curr)
    {
        DEBUG_PROFILE_SCOPE("Octree::updateOctant");

        if(oct->isLeaf())
        {
            for(const auto& pt:points)
            {
                bool fused = false;

                for(auto* g : oct->gaussians)
                {
                    if(g->checkFit(pt.pos, pt.cov, chi_threshold_))
                    {
                        g->fuseAdaptive(pt, P_curr);
                        fused = true;
                        break;
                    }
                }

                if(!fused)
                {
                    oct->gaussians.push_back(allocGaussian(pt));
                    num_points_++;
                }
            }

            merge(oct);

            if(oct->gaussians.size() > max_gaussians_leaf_
               && oct->extent > 2*min_extent_)
            {
                split(oct);
            }

            return;
        }

        std::vector<VecPointCov> child_pts(8);

        for(const auto& p:points)
            child_pts[getIdx(p.pos,oct->centroid)].push_back(p);

        for(int i=0;i<8;i++)
        {
            if(child_pts[i].empty()) continue;

            if(!oct->children[i])
            {
                Point c = computeChildCenter(oct->centroid, oct->extent,i);

                oct->children[i] =
                    createOctant(c, 0.5*oct->extent, child_pts[i]);
            }
            else
            {
                updateOctant(oct->children[i], child_pts[i], P_curr);
            }
        }
    }

    /*
    ========================================
    SPLIT
    ========================================
    */

    void split(Octant* oct)
    {
        DEBUG_PROFILE_SCOPE("Octree::split");

        oct->init_children();

        std::vector<std::vector<Gaussian*>> child_gaussians(8);

        // distribute gaussians to children
        for(auto* g : oct->gaussians)
        {
            int idx = getIdx(g->mean, oct->centroid);
            child_gaussians[idx].push_back(g);
        }

        // create children
        for(int i = 0; i < 8; i++)
        {
            if(child_gaussians[i].empty())
                continue;

            Point child_center =
                computeChildCenter(oct->centroid, oct->extent, i);

            Octant* child = allocOctant();

            child->centroid = child_center;
            child->extent = 0.5 * oct->extent;

            child->gaussians = std::move(child_gaussians[i]);

            oct->children[i] = child;
        }

        // parent is no longer leaf
        oct->gaussians.clear();
    }

    /*
    ========================================
    GAUSSIAN MERGING
    ========================================
    */

    void merge(Octant*& oct)
    {
        DEBUG_PROFILE_SCOPE("Octree::merge");

        // To-Do: sort gaussians to avoid O(n^2)
        for(size_t i=0; i < oct->gaussians.size(); i++)
        {
            for(size_t j=i+1; j < oct->gaussians.size();)
            {
                Gaussian* gi = oct->gaussians[i];
                Gaussian* gj = oct->gaussians[j];

                if(gi->checkFit(gj->mean, gj->cov, chi_threshold_))
                {
                    gi->merge(*gj);
                    
                    gaussian_pool_.free(gj); // free the merged Gaussian from pool
                    oct->gaussians.erase(oct->gaussians.begin()+j); 
                }
                else j++;
            }
        }
    }

    /*
    ========================================
    SEARCH IMPLEMENTATION
    ========================================
    */

    bool knnRecursive(const Octant* oct, const Point& q, NeighborHeap& heap) const
    {
        DEBUG_PROFILE_SCOPE("Octree::knnRecursive");

        if(!oct) return false;

        if(oct->isLeaf())
        {
            for(auto* g : oct->gaussians)
            {
                Scalar d2 = (g->mean - q).squaredNorm();
                heap.add(*g, d2);
            }
            return heap.full() && isInsideOctant(oct, q, heap.worstDist());
        }

        int morton = getIdx(q, oct->centroid);
        if(knnRecursive(oct->children[morton], q, heap)) return true;

        static const int ordered_indices[8][7] = {
            {1, 2, 4, 3, 5, 6, 7}, {0, 3, 5, 2, 4, 7, 6}, {0, 3, 6, 1, 4, 7, 5}, {1, 2, 7, 0, 5, 6, 4},
            {0, 5, 6, 1, 2, 7, 3}, {1, 4, 7, 0, 3, 6, 2}, {2, 4, 7, 0, 3, 5, 1}, {3, 5, 6, 1, 2, 4, 0}
        };

        for(int i=0;i<7;i++)
        {
            int c = ordered_indices[morton][i];
            if(!oct->children[c]) continue;
            if(heap.full() && !overlaps(oct->children[c], q, heap.worstDist())) continue;
            if(knnRecursive(oct->children[c], q, heap)) return true;
        }

        return heap.full() && isInsideOctant(oct, q, heap.worstDist());
    }

    void radiusRecursive(const Octant* oct,
                         const Point& q,
                         Scalar sqr_radius,
                         std::vector<Gaussian*>& neighbors,
                         std::vector<Scalar>& distances) const
    {
        DEBUG_PROFILE_SCOPE("Octree::radiusRecursive");

        if(!oct) return;

        if(oct->isLeaf())
        {
            for(auto* g : oct->gaussians)
            {
                Scalar d2 = (g->mean - q).squaredNorm();
                if(d2 <= sqr_radius)
                {
                    neighbors.push_back(g);
                    distances.push_back(std::sqrt(d2));
                }
            }
            return;
        }

        for(int i=0;i<8;i++)
        {
            if(!oct->children[i] || !overlaps(oct->children[i], q, sqr_radius)) continue;
            radiusRecursive(oct->children[i], q, sqr_radius, neighbors, distances);
        }
    }

    /*
    ========================================
    UTILS
    ========================================
    */

    static int getIdx(const Point& p,const Point& c)
    {
        return (p.x() > c.x() ? 1 : 0)
             | (p.y() > c.y() ? 2 : 0)
             | (p.z() > c.z() ? 4 : 0);
    }

    static Point computeChildCenter(const Point& c,
                                    Scalar extent,
                                    int idx)
    {
        static const Scalar factor[2]={-0.5,0.5};

        return Point(
            c.x() + factor[(idx & 1) > 0] * extent,
            c.y() + factor[(idx & 2) > 0] * extent,
            c.z() + factor[(idx & 4) > 0] * extent
        );
    }

    bool overlaps(const Octant* oct, const Point& q, Scalar sqr_radius) const
    {
        Point dist = ((q - oct->centroid).cwiseAbs().array() - oct->extent);
    
        if ((dist.x() > 0 and dist.x()*dist.x() > sqr_radius) or
            (dist.y() > 0 and dist.y()*dist.y() > sqr_radius) or
            (dist.z() > 0 and dist.z()*dist.z() > sqr_radius))
        return false;

        int num_less_extent = (dist.x() < 0) + (dist.y() < 0) + (dist.z() < 0);

        if (num_less_extent > 1) return true;

        return (dist.cwiseMax(0.0).squaredNorm() < sqr_radius);
    }

    bool isInsideOctant(const Octant* oct, const Point& q, Scalar radius) const
    {
        Point dist = oct->extent - (q - oct->centroid).cwiseAbs().array();
    
        return (dist.x() < 0 or dist.x()*dist.x() < radius) ? false : 
                (dist.y() < 0 or dist.y()*dist.y() < radius) ? false :
                (dist.z() < 0 or dist.z()*dist.z() < radius) ? false :
                true;
    }

    void freeSubtree(Octant* oct)
    {
        if(!oct) return;

        for(int i=0;i<8;i++)
        {
            if(oct->children[i])
                freeSubtree(oct->children[i]);
        }

        octant_pool_.free(oct);
    }
};

} // namespace gauss_mapping