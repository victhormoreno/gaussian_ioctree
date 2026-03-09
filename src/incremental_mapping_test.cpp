// gaussian_octree_test.cpp
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

class GaussianOctreeIncrementalMapping : public rclcpp::Node {
public:
  GaussianOctreeIncrementalMapping() : Node("gaussian_octree_test")
  {
    // ---- params (only these) ----
    pcd_path_  = this->declare_parameter<std::string>("pcd_path", "");
    leaf_size_ = this->declare_parameter<double>("leaf_size", 1.0);
    chi_       = this->declare_parameter<double>("chi", 1.815);
    filter_x_  = this->declare_parameter<double>("filter_x", 5.0);
    batch_size_ = this->declare_parameter<int>("batch_size", 2000);

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

    p_cov_ = 0.01 * gauss_mapping::Mat3::Identity();
    P_curr_ = p_cov_;

    const size_t max_g = 5;
    const gauss_mapping::Scalar min_extent = static_cast<gauss_mapping::Scalar>(leaf_size_);

    octree_ = std::make_unique<gauss_mapping::Octree>(
        max_g,
        min_extent,
        static_cast<gauss_mapping::Scalar>(chi_));

    insert_index_ = 0;

    octree_loaded_ = true;

    RCLCPP_INFO(get_logger(), "Octree initialized. Ready for incremental insertion.");
  }

  void incremental_update()
  {
    if(insert_index_ >= filtered_cloud_->size())
        return;

    gauss_mapping::VecPointCov batch;
    batch.reserve(batch_size_);

    size_t end = std::min(insert_index_ + batch_size_,
                          filtered_cloud_->size());

    for(size_t i = insert_index_; i < end; ++i)
    {
      const auto& p = filtered_cloud_->points[i];

      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
          continue;

      gauss_mapping::Point pos;
      pos << p.x, p.y, p.z;

      batch.emplace_back(pos, p_cov_);
    }

    auto tick = std::chrono::steady_clock::now();
    octree_->update(batch, P_curr_);
    auto tack = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = tack - tick;

    insert_index_ = end;

    cached_gaussians_ = octree_->getGaussians();

    RCLCPP_INFO(get_logger(),
        "Inserted %zu/%zu points | Gaussians: %zu (%zu points) | %.3f ms",
        insert_index_,
        filtered_cloud_->size(),
        octree_->size(),
        octree_->num_points(),
        elapsed.count()*1000.0);
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

    const size_t n = cached_gaussians_.size();
    arr.markers.reserve(n + 1);

    for (size_t i = 0; i < n; ++i) {
      const auto& g = *cached_gaussians_[i];

      gauss_mapping::Mat3 Sigma = g.cov;

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

      m.lifetime = rclcpp::Duration(1.0, 0);
      arr.markers.push_back(m);
    }

    return arr;
  }

  void publish_once()
  {
    if (!cloud_loaded_ || !octree_loaded_)
        return;

    incremental_update();

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

  size_t insert_index_{0};
  size_t batch_size_{2000};   // points per update step

  gauss_mapping::Mat3 p_cov_;
  gauss_mapping::Mat3 P_curr_;

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
    auto node = std::make_shared<GaussianOctreeIncrementalMapping>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("gaussian_octree_test"), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}