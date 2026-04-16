// real_time_test_ivox.cpp
//
// Publishes at LiDAR frequency:
//  - Filtered PCD cloud on /pcd
//  - Gaussian ellipsoids on /gaussians
//
// Params:
//  - lidar_topic (string)
//  - res (double)
//  - update_thresh (int)

#include <chrono>
#include <string>
#include <stdexcept>
#include <memory>
#include <limits>
#include <algorithm>
#include <cmath>
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#define PCL_NO_PRECOMPILE
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/pcl_config.h>

#include <Eigen/Dense>
// #include "gaussian_octree/gauss_ivox_deprecated.hpp"
#include "gaussian_octree/gauss_ivox.hpp"

struct PointType {
    PointType(): data{0.f, 0.f, 0.f, 1.f} {}
    PointType(float x, float y, float z): data{x, y, z, 1.f} {}

    PCL_ADD_POINT4D;
    float intensity;
    union {
      std::uint32_t t;   // (Ouster) time since beginning of scan in nanoseconds
      float time;        // (Velodyne) time since beginning of scan in seconds
      double timestamp;  // (Hesai) absolute timestamp in seconds
                         // (Livox) absolute timestamp in (seconds * 10e9)
    };
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;


POINT_CLOUD_REGISTER_POINT_STRUCT(PointType,
                                 (float, x, x)
                                 (float, y, y)
                                 (float, z, z)
                                 (float, intensity, intensity)
                                 (std::uint32_t, t, t)
                                 (float, time, time)
                                 (double, timestamp, timestamp))

using namespace std::chrono_literals;

class GaussianIVoxRealTime : public rclcpp::Node {
public:
  GaussianIVoxRealTime() : Node("gaussian_ivox_test")
  {
    // ---- params (only these) ----
    lidar_topic_      = this->declare_parameter<std::string>("lidar_topic", "/lidar_points");
    resolution_       = this->declare_parameter<double>("res", 2.0);
    update_threshold_ = this->declare_parameter<int>("update_thresh", 5);

    // fixed topics/frames
    frame_id_ = "map";
    topic_    = "/pcd";
    gaussian_topic_ = "/gaussians";
    voxel_topic_ = "/voxels";

    if (resolution_ <= 0.0) resolution_ = 0.001;
    if (update_threshold_ <= 0) update_threshold_ = 5;

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::QoS(1).transient_local());
    gauss_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      gaussian_topic_, rclcpp::QoS(1).transient_local());
    voxel_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      voxel_topic_, rclcpp::QoS(1).transient_local());

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    lidar_topic_, 1, std::bind(&GaussianIVoxRealTime::lidar_callback, this, std::placeholders::_1));

    // Declare Incremental Voxel
    ivox_ = std::make_unique<gauss_ivox_mapping::GaussianIVox>(
      static_cast<gauss_ivox_mapping::Scalar>(resolution_),
      static_cast<std::size_t>(update_threshold_)
    );

    RCLCPP_INFO(get_logger(),
                "Ready: topic='%s' res=%.3f update_thresh=%d",
                lidar_topic_.c_str(), resolution_, update_threshold_);
  }

private:
  void lidar_callback(const sensor_msgs::msg::PointCloud2 & msg) {

    pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
    pcl::fromROSMsg(msg, *pc_);

    auto tick = std::chrono::system_clock::now(); 
    for (const auto& p : pc_->points) {
      gauss_ivox_mapping::Point point;
      point << static_cast<gauss_ivox_mapping::Scalar>(p.x),
                    static_cast<gauss_ivox_mapping::Scalar>(p.y),
                    static_cast<gauss_ivox_mapping::Scalar>(p.z);
      ivox_->update(point);
    }
    auto tack = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_time = tack-tick;

    // RCLCPP_INFO(get_logger(), "Ivox contains %zu gaussians (%zu points) and took %f ms to update",
    //             ivox_->size(), ivox_->num_points(), elapsed_time.count()*1000.0);

    cloud_pub_->publish(msg);


    std::vector<gauss_ivox_mapping::GaussPtr> gauss = ivox_->getGaussians();

    RCLCPP_INFO(get_logger(), "Ivox has %zu active gaussians and took %f ms to update", gauss.size(), elapsed_time.count()*1000.0);

    gauss_pub_->publish(makeGaussianMarkers(*ivox_.get(), this->now()));
    voxel_pub_->publish(makeVoxelMarkers(*ivox_.get(), this->now(), resolution_));

  }

  visualization_msgs::msg::MarkerArray makeGaussianMarkers(
      const gauss_ivox_mapping::GaussianIVox& ivox,
      const rclcpp::Time& stamp)
  {
      visualization_msgs::msg::MarkerArray arr;
      std::shared_lock lock(ivox.map_mtx_); // safe map access

      std::mt19937 gen(12345); // fixed seed for consistent cluster colors
      std::uniform_real_distribution<float> dis(0.0f, 1.0f);
      std::unordered_map<gauss_ivox_mapping::UnionFindNode*, std::tuple<float,float,float>> parent_colors;
      std::unordered_map<gauss_ivox_mapping::UnionFindNode*, std::vector<Eigen::Vector3i>> clusters;

      // Build clusters based on Union-Find parents and assign colors
      for (const auto& [key, node] : ivox.map_) {
          auto root = node->find();
          clusters[root].push_back(key);
          if (parent_colors.find(root) == parent_colors.end()) {
              parent_colors[root] = {dis(gen), dis(gen), dis(gen)};
          }
      }

      // Visualize each voxel Gaussian primitive
      int id = 0;
      for (auto& [root, voxels] : clusters) {

          auto& gaus = root->gauss_ptr;
          if (!gaus) continue; // safety check, should always have a Gaussian

          for (const auto& key : voxels) {

            gauss_ivox_mapping::Point voxel_center;
            voxel_center[0] = (key.x() + 0.5) * resolution_;
            voxel_center[1] = (key.y() + 0.5) * resolution_;
            voxel_center[2] = (key.z() + 0.5) * resolution_;

            visualization_msgs::msg::Marker m;

            switch (gaus->type) {
                case gauss_ivox_mapping::PrimitiveType::PLANE:
                    m = makePlaneMarker(*gaus, voxel_center, id++, stamp, parent_colors[root]);
                    break;

                case gauss_ivox_mapping::PrimitiveType::VOLUME:
                    m = makeVolumeMarker(*gaus, id++, stamp, parent_colors[root]);
                    break;

                default:
                    break;
            }

            arr.markers.push_back(m);
        }

      }
      
      return arr;
  }

  visualization_msgs::msg::Marker makePlaneMarker(
      const gauss_ivox_mapping::GaussianPrimitive& g,
      const gauss_ivox_mapping::Point& voxel_center,
      int id,
      const rclcpp::Time& stamp,
      std::tuple<float,float,float>& color) const
  {

      using Point = gauss_ivox_mapping::Point;

      visualization_msgs::msg::Marker m;

      const double plane_size = resolution_;
      const double thickness = 0.002;

      auto buildOmega = [](const Point& p, int main_dir) {
        Point w;
        switch (main_dir) {
            case 0: w << p[0], p[1], 1.0; break;
            case 1: w << p[0], 1.0, p[1]; break;
            case 2: w << 1.0, p[0], p[1]; break;
        }
        return w;
      };

      // Compute plane center and normal
      Point normal = buildOmega(g.param, g.main_dir);
      double norm2 = normal.squaredNorm();
      Point p0 = voxel_center; 

      Point p_plane = p0 - normal * (normal.dot(p0) + g.param[2]) / norm2;

      normal.normalize();
      if (normal.norm() < 1e-6) normal = Eigen::Vector3d::UnitZ();

      Eigen::Vector3d z_axis = normal;

      // Pick a WORLD axis that is least aligned with the normal
      Eigen::Vector3d ref_axis;
      if (std::abs(z_axis.x()) <= std::abs(z_axis.y()) &&
          std::abs(z_axis.x()) <= std::abs(z_axis.z())) {
          ref_axis = Eigen::Vector3d::UnitX();
      }
      else if (std::abs(z_axis.y()) <= std::abs(z_axis.z())) {
          ref_axis = Eigen::Vector3d::UnitY();
      }
      else {
          ref_axis = Eigen::Vector3d::UnitZ();
      }

      // Build orthonormal basis
      Eigen::Vector3d x_axis = ref_axis.cross(z_axis).normalized();
      Eigen::Vector3d y_axis = z_axis.cross(x_axis).normalized();

      Eigen::Matrix3d R;
      R.col(0) = x_axis;
      R.col(1) = y_axis;
      R.col(2) = normal;

      Eigen::Quaterniond q(R);

      m.header.frame_id = frame_id_;
      m.header.stamp = stamp;
      m.ns = "planes";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::CUBE;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = p_plane.x();
      m.pose.position.y = p_plane.y();
      m.pose.position.z = p_plane.z();

      m.pose.orientation.x = q.x();
      m.pose.orientation.y = q.y();
      m.pose.orientation.z = q.z();
      m.pose.orientation.w = q.w();

      m.scale.x = plane_size; 
      m.scale.y = plane_size;
      m.scale.z = thickness;
      // m.scale.x = plane_size * sqrt(g.cov(0,0)); // spread along x-axis
      // m.scale.y = plane_size * sqrt(g.cov(1,1)); // spread along y-axis
      // m.scale.z = sqrt(g.cov(2,2)); // spread along z-axis

      m.color.r = std::get<0>(color);
      m.color.g = std::get<1>(color);
      m.color.b = std::get<2>(color);
      m.color.a = 0.8f;

      m.lifetime = rclcpp::Duration(0, 0);

      return m;
  }

  visualization_msgs::msg::Marker makeVolumeMarker(
      const gauss_ivox_mapping::GaussianPrimitive& g,
      int id,
      const rclcpp::Time& stamp,
      std::tuple<float,float,float>& color
    ) const
  {
      visualization_msgs::msg::Marker m;

      const double k_sigma = 1.0;

      Eigen::SelfAdjointEigenSolver<gauss_ivox_mapping::Mat3> es(g.cov);
      if (es.info() != Eigen::Success) return m;

      Eigen::Vector3d evals = es.eigenvalues();
      Eigen::Matrix3d evecs = es.eigenvectors();

      constexpr double EPS = 1e-12;
      evals = evals.cwiseMax(EPS);

      if (evecs.determinant() < 0.0)
          evecs.col(0) *= -1.0;

      Eigen::Quaterniond q(evecs);

      m.header.frame_id = frame_id_;
      m.header.stamp = stamp;
      m.ns = "volumes";
      m.id = id;
      m.type = visualization_msgs::msg::Marker::SPHERE;
      m.action = visualization_msgs::msg::Marker::ADD;

      m.pose.position.x = g.mean.x();
      m.pose.position.y = g.mean.y();
      m.pose.position.z = g.mean.z();

      m.pose.orientation.x = q.x();
      m.pose.orientation.y = q.y();
      m.pose.orientation.z = q.z();
      m.pose.orientation.w = q.w();

      m.scale.x = 2.0 * k_sigma * std::sqrt(evals(0));
      m.scale.y = 2.0 * k_sigma * std::sqrt(evals(1));
      m.scale.z = 2.0 * k_sigma * std::sqrt(evals(2));

      m.color.r = std::get<0>(color);
      m.color.g = std::get<1>(color);
      m.color.b = std::get<2>(color);
      m.color.a = 0.8f;

      m.lifetime = rclcpp::Duration(0, 0);

      return m;
  }

  visualization_msgs::msg::MarkerArray makeVoxelMarkers(
      const gauss_ivox_mapping::GaussianIVox& ivox,
      const rclcpp::Time& stamp,
      double voxel_size)
  {
      visualization_msgs::msg::MarkerArray arr;
      std::shared_lock lock(ivox.map_mtx_); // safe map access

      std::mt19937 gen(12345); // fixed seed for consistent cluster colors
      std::uniform_real_distribution<float> dis(0.0f, 1.0f);
      std::unordered_map<gauss_ivox_mapping::UnionFindNode*, std::tuple<float,float,float>> parent_colors;

      int id = 0;
      for (const auto& [key, node] : ivox.map_) {
          auto* root = node->find();
          if (parent_colors.find(root) == parent_colors.end()) {
              parent_colors[root] = {dis(gen), dis(gen), dis(gen)};
          }
          auto [r,g,b] = parent_colors[root];

          gauss_ivox_mapping::Point voxel_center_;
          voxel_center_[0] = (key.x() + 0.5) * voxel_size;
          voxel_center_[1] = (key.y() + 0.5) * voxel_size;
          voxel_center_[2] = (key.z() + 0.5) * voxel_size;

          // --- Voxel cube ---
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "map";
          m.header.stamp = stamp;
          m.ns = "voxels";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::CUBE;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = voxel_center_[0];
          m.pose.position.y = voxel_center_[1];
          m.pose.position.z = voxel_center_[2];
          m.pose.orientation.w = 1.0;
          m.scale.x = voxel_size;
          m.scale.y = voxel_size;
          m.scale.z = voxel_size;
          m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 0.2f;
          arr.markers.push_back(m);

          // --- Optional: Add text marker with point count ---
      }
      
      return arr;
  }

private:
  // params
  double resolution_{5.0};
  int update_threshold_{5};

  // fixed
  std::string lidar_topic_{""};
  std::string frame_id_{"map"};
  std::string topic_{"/pcd"};
  std::string gaussian_topic_{"/gaussians"};
  std::string voxel_topic_{"/voxels"};

  // map
  std::unique_ptr<gauss_ivox_mapping::GaussianIVox> ivox_{nullptr};

  // ros
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gauss_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<GaussianIVoxRealTime>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("gaussian_ivox_test"), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}