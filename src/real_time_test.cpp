// gaussian_octree_test.cpp
//
// Publishes at 1 Hz:
//  - Filtered PCD cloud on /pcd
//  - Gaussian ellipsoids on /gaussians
//
// Params:
//  - lidar_topic (string)
//  - leaf_size (double)
//  - chi (double)

#include <chrono>
#include <string>
#include <stdexcept>
#include <memory>
#include <limits>
#include <algorithm>
#include <cmath>

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
#include "gaussian_octree/gauss_octree.hpp"

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

class GaussianOctreeRealTime : public rclcpp::Node {
public:
  GaussianOctreeRealTime() : Node("gaussian_octree_test")
  {
    // ---- params (only these) ----
    lidar_topic_  = this->declare_parameter<std::string>("lidar_topic", "/lidar_points");
    leaf_size_ = this->declare_parameter<double>("leaf_size", 1.0);
    chi_       = this->declare_parameter<double>("chi", 7.815);

    // fixed topics/frames
    frame_id_ = "map";
    topic_    = "/pcd";
    gaussian_topic_ = "/gaussians";

    if (leaf_size_ <= 0.0) leaf_size_ = 1e-6;
    if (chi_ <= 0.0) chi_ = 1e-6;

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::QoS(1).transient_local());
    gauss_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      gaussian_topic_, rclcpp::QoS(1).transient_local());
    gaussian_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      "/gaussian_cloud", rclcpp::QoS(1).transient_local());

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    lidar_topic_, 1, std::bind(&GaussianOctreeRealTime::lidar_callback, this, std::placeholders::_1));

    // Declare Octree
    const size_t max_g = 5;
    const gauss_mapping::Scalar min_extent = static_cast<gauss_mapping::Scalar>(leaf_size_);

    octree_ = std::make_unique<gauss_mapping::Octree>(
      max_g,
      min_extent,
      static_cast<gauss_mapping::Scalar>(chi_)
    );

    RCLCPP_INFO(get_logger(),
                "Ready: topic='%s' leaf_size=%.6f chi=%.3f (publishing 1 Hz)",
                lidar_topic_.c_str(), leaf_size_, chi_);
  }

private:
  void lidar_callback(const sensor_msgs::msg::PointCloud2 & msg) {

    pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
    pcl::fromROSMsg(msg, *pc_);

    // point covariance
    const gauss_mapping::Mat3 p_cov = 0.01 * gauss_mapping::Mat3::Identity();
    const gauss_mapping::Mat3 P_curr = p_cov; // fixed eKF covariance

    gauss_mapping::VecPointCov batch;
    batch.reserve(pc_->points.size());
    for (const auto& p : pc_->points) {
      gauss_mapping::Point pos;
      pos << static_cast<gauss_mapping::Scalar>(p.x),
             static_cast<gauss_mapping::Scalar>(p.y),
             static_cast<gauss_mapping::Scalar>(p.z);
      batch.emplace_back(pos, p_cov);
    }

    auto tick = std::chrono::system_clock::now(); 
    octree_->update(batch, P_curr);
    auto tack = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_time = tack-tick;

    RCLCPP_INFO(get_logger(), "Octree contains %zu gaussians (num_points=%zu) and took %f ms to update",
                octree_->size(), octree_->num_points(), elapsed_time.count()*1000.0);

    cached_gaussians_.clear();
    cached_gaussians_ = octree_->getGaussians();

    cloud_pub_->publish(msg);

    gaussian_cloud_pub_->publish(make_gaussian_cloud(this->now()));

    gauss_pub_->publish(make_gaussian_markers(this->now()));

  }
  
  sensor_msgs::msg::PointCloud2 make_gaussian_cloud(const rclcpp::Time& stamp) const
  {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    cloud.reserve(cached_gaussians_.size());

    for (const auto* g : cached_gaussians_) {
      pcl::PointXYZI p;
      p.x = g->mean.x();
      p.y = g->mean.y();
      p.z = g->mean.z();
      p.intensity = std::sqrt(g->getCovariance().trace()); // geometric covariance
      // p.intensity = g->count; // points count
      cloud.push_back(p);
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);

    msg.header.frame_id = frame_id_;
    msg.header.stamp = stamp;

    return msg;
  }

  visualization_msgs::msg::MarkerArray make_gaussian_markers(const rclcpp::Time& stamp) const {
    visualization_msgs::msg::MarkerArray arr;

    visualization_msgs::msg::Marker del;
    del.header.frame_id = frame_id_;
    del.header.stamp = stamp;
    del.ns = "gaussians";
    del.id = 0;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    const double k_sigma = 1.0;
    const double alpha   = 0.7;
    const size_t max_markers = 5000;

    const size_t n = std::min(max_markers, cached_gaussians_.size());
    arr.markers.reserve(n + 1);

    for (size_t i = 0; i < n; ++i) {
      const auto& g = *cached_gaussians_[n-1-i];

      gauss_mapping::Mat3 Sigma = g.cov; // not normalized covariance

      Eigen::SelfAdjointEigenSolver<gauss_mapping::Mat3> es(Sigma);
      if (es.info() != Eigen::Success) continue;

      Eigen::Vector3d evals = es.eigenvalues();
      Eigen::Matrix3d evecs = es.eigenvectors();

      constexpr double EPS = 1e-12;
      evals(0) = std::max(evals(0), EPS);
      evals(1) = std::max(evals(1), EPS);
      evals(2) = std::max(evals(2), EPS);
      if (evecs.determinant() < 0.0) evecs.col(0) *= -1.0;

      Eigen::Quaterniond q(evecs);

      visualization_msgs::msg::Marker m;
      m.header.frame_id = frame_id_;
      m.header.stamp = stamp;
      m.ns = "gaussians";
      m.id = static_cast<int>(i + 1);
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

      // Example: color by count (or any Gaussian property)
      float k = 10.0f; // controls saturation speed -> k points is half the colormap value
      float v = float(g.count) / (float(g.count) + k);
      m.color.r = v;
      m.color.g = 1.0f - std::fabs(v - 0.5f) * 2.0f;
      m.color.b = 1.0f - v;

      m.color.a = static_cast<float>(alpha);

      m.lifetime = rclcpp::Duration(0, 0);
      arr.markers.push_back(m);
    }

    return arr;
  }

private:
  // params
  std::string pcd_path_;
  double leaf_size_{1.0};
  double chi_{7.815};

  // fixed
  std::string lidar_topic_{""};
  std::string frame_id_{"map"};
  std::string topic_{"/pcd"};
  std::string gaussian_topic_{"/gaussians"};

  // map
  std::unique_ptr<gauss_mapping::Octree> octree_{nullptr};
  std::vector<gauss_mapping::Gaussian*> cached_gaussians_;

  // ros
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr gaussian_cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gauss_pub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<GaussianOctreeRealTime>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("gaussian_octree_test"), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}