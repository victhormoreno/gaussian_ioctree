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
#include "gaussian_octree/octree.hpp"

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
    lidar_topic_ = this->declare_parameter<std::string>("lidar_topic", "/lidar_points");
    double min_extent_d = this->declare_parameter<double>("min_extent", 2.0);
    bucket_size_ = this->declare_parameter<int>("bucket_size", 5);

    min_extent_ = static_cast<float>(min_extent_d);

    lidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    lidar_topic_, 1, std::bind(&GaussianIVoxRealTime::lidar_callback, this, std::placeholders::_1));

    // Init Octree
    octree_ = std::make_unique<octree::Octree>();
    octree_->setBucketSize(bucket_size_);
    octree_->setDownsample(false);
    octree_->setMinExtent(min_extent_);

    RCLCPP_INFO(get_logger(),
                "Ready: topic='%s' min_extent=%.3f bucket_size=%d",
                lidar_topic_.c_str(), min_extent_d, bucket_size_);
  }

private:
  void lidar_callback(const sensor_msgs::msg::PointCloud2 & msg) {

    pcl::PointCloud<PointType>::Ptr pc_ (std::make_shared<pcl::PointCloud<PointType>>());
    pcl::fromROSMsg(msg, *pc_);

    auto tick = std::chrono::system_clock::now(); 
    if(octree_->num_points_ > 0)
        octree_->update(pc_);
    else 
        octree_->initialize(pc_);
    auto tack = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_time = tack-tick;

    RCLCPP_INFO(get_logger(), "Octree took %f ms to update", elapsed_time.count()*1000.0);

    RCLCPP_INFO(this->get_logger(), 
                octree_->profiler_.str()
                .c_str());

  }

  private:
  // params
  int bucket_size_;
  float min_extent_;

  // fixed
  std::string lidar_topic_{""};

  // map
  std::unique_ptr<octree::Octree > octree_{nullptr};

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