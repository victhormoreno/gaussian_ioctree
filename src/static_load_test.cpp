// static_load_test.cpp
//
// Publishes at 1 Hz:
//  - Filtered PCD cloud on /pcd
//  - Gaussian ellipsoids on /gaussians
//
// Params:
//  - pcd_path (string)
//  - leaf_size (double)
//  - chi (double)
//  - filter_x (double)         // x,y in [-filter_x, +filter_x]

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

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
#include "gaussian_octree/gauss_octree.hpp"

using namespace std::chrono_literals;

class GaussianOctreeStaticLoad : public rclcpp::Node {
public:
  GaussianOctreeStaticLoad() : Node("gaussian_octree_test")
  {
    // ---- params (only these) ----
    pcd_path_  = this->declare_parameter<std::string>("pcd_path", "");
    leaf_size_ = this->declare_parameter<double>("leaf_size", 1.0);
    chi_       = this->declare_parameter<double>("chi", 7.815);
    filter_x_  = this->declare_parameter<double>("filter_x", 10.0);

    // fixed topics/frames
    frame_id_ = "map";
    topic_    = "/pcd";
    gaussian_topic_ = "/gaussians";

    if (pcd_path_.empty()) throw std::runtime_error("pcd_path is empty");
    if (leaf_size_ <= 0.0) leaf_size_ = 1e-6;
    if (chi_ <= 0.0) chi_ = 1e-6;
    if (filter_x_ <= 0.0) filter_x_ = 1e-6;

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::QoS(1).transient_local());
    gauss_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      gaussian_topic_, rclcpp::QoS(1).transient_local());

    load_and_filter_once();
    build_octree_once();

    // Publish periodically (1 Hz)
    timer_ = this->create_wall_timer(1s, [this]() { publish_once(); });

    RCLCPP_INFO(get_logger(),
                "Ready: pcd='%s' leaf_size=%.6f chi=%.3f filter_x=%.3f (publishing 1 Hz)",
                pcd_path_.c_str(), leaf_size_, chi_, filter_x_);
  }

private:
  void load_and_filter_once() {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    int ret = pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_path_, *cloud);
    if (ret < 0) throw std::runtime_error("Failed to load PCD");

    // x in [-x, +x]
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_x(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PassThrough<pcl::PointXYZI> pass_x;
    pass_x.setInputCloud(cloud);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(static_cast<float>(-filter_x_), static_cast<float>(+filter_x_));
    pass_x.filter(*cloud_x);

    // y in [-x, +x]
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xy(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PassThrough<pcl::PointXYZI> pass_y;
    pass_y.setInputCloud(cloud_x);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(static_cast<float>(-filter_x_), static_cast<float>(+filter_x_));
    pass_y.filter(*cloud_xy);

    filtered_cloud_ = cloud_xy;

    pcl::toROSMsg(*filtered_cloud_, cloud_msg_);
    cloud_msg_.header.frame_id = frame_id_;
    cloud_loaded_ = true;

    RCLCPP_INFO(get_logger(), "PCD: %zu -> %zu points (filter_x=%.3f)",
                cloud->size(), filtered_cloud_->size(), filter_x_);
  }

  void build_octree_once() {
    if (!filtered_cloud_ || filtered_cloud_->empty())
      throw std::runtime_error("Filtered cloud is empty");

    // point covariance
    const gauss_mapping::Mat3 p_cov = 0.01 * gauss_mapping::Mat3::Identity();
    const gauss_mapping::Mat3 P_curr = p_cov;

    // internal constants
    const size_t max_g = 5;
    const gauss_mapping::Scalar min_extent = static_cast<gauss_mapping::Scalar>(leaf_size_);

    octree_ = std::make_unique<gauss_mapping::Octree>(
      max_g,
      min_extent,
      static_cast<gauss_mapping::Scalar>(chi_)
    );

    gauss_mapping::VecPointCov batch;
    // batch.reserve(finite_count);
    for (const auto& p : filtered_cloud_->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
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
    
    octree_loaded_ = true;

    cached_gaussians_.clear();
    cached_gaussians_ = octree_->getGaussians();

    RCLCPP_INFO(get_logger(), "Octree contains %zu gaussians (num_points=%zu) and took %f ms to load",
                octree_->size(), octree_->num_points(), elapsed_time.count()*1000.0);
  }

  visualization_msgs::msg::MarkerArray make_gaussian_markers(const rclcpp::Time& stamp) const {
    visualization_msgs::msg::MarkerArray arr;
    if (!octree_loaded_) return arr;

    visualization_msgs::msg::Marker del;
    del.header.frame_id = frame_id_;
    del.header.stamp = stamp;
    del.ns = "gaussians";
    del.id = 0;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    const double k_sigma = 1.0;
    const double alpha   = 0.7;
    const size_t max_markers = 50000;

    const size_t n = std::min(max_markers, cached_gaussians_.size());
    arr.markers.reserve(n + 1);

    for (size_t i = 0; i < n; ++i) {
      const auto& g = *cached_gaussians_[i];

      gauss_mapping::Mat3 Sigma = g.getCovariance();

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

      m.color.r = 0.2f;
      m.color.g = 0.9f;
      m.color.b = 0.9f;
      m.color.a = static_cast<float>(alpha);

      m.lifetime = rclcpp::Duration(0, 0);
      arr.markers.push_back(m);
    }

    return arr;
  }

  void publish_once() {
    if (!cloud_loaded_ || !octree_loaded_) return;
    const auto stamp = this->now();

    cloud_msg_.header.stamp = stamp;
    cloud_pub_->publish(cloud_msg_);

    gauss_pub_->publish(make_gaussian_markers(stamp));
  }

private:
  // params
  std::string pcd_path_;
  double leaf_size_{1.0};
  double chi_{7.815};
  double filter_x_{10.0};

  // fixed
  std::string frame_id_{"map"};
  std::string topic_{"/pcd"};
  std::string gaussian_topic_{"/gaussians"};

  // state
  bool cloud_loaded_{false};
  bool octree_loaded_{false};
  sensor_msgs::msg::PointCloud2 cloud_msg_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_{nullptr};

  std::unique_ptr<gauss_mapping::Octree> octree_{nullptr};
  std::vector<gauss_mapping::Gaussian*> cached_gaussians_;

  // ros
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gauss_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<GaussianOctreeStaticLoad>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("gaussian_octree_test"), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}