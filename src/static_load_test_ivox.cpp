// static_load_test_ivox.cpp
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
#include <random>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl_conversions/pcl_conversions.h>

#include <Eigen/Dense>
// #include "gaussian_octree/gauss_ivox_deprecated.hpp"
#include "gaussian_octree/gauss_ivox.hpp"

using namespace std::chrono_literals;

class GaussianIVOXStaticLoad : public rclcpp::Node {
public:
  GaussianIVOXStaticLoad() : Node("gaussian_ivox_test")
  {
    // ---- params (only these) ----
    pcd_path_         = this->declare_parameter<std::string>("pcd_path", "");
    resolution_       = this->declare_parameter<double>("res", 2.0);
    update_threshold_ = this->declare_parameter<int>("update_thresh", 5);
    filter_x_         = this->declare_parameter<double>("filter_x", 10.0);

    // fixed topics/frames
    frame_id_ = "map";
    topic_    = "/pcd";
    gaussian_topic_ = "/gaussians";
    voxel_topic_ = "/voxels";

    if (pcd_path_.empty()) throw std::runtime_error("pcd_path is empty");
    if (resolution_ <= 0.0) resolution_ = 0.001;
    if (update_threshold_ <= 0) update_threshold_ = 5;
    if (filter_x_ <= 0.0) filter_x_ = 0.001;

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      topic_, rclcpp::QoS(1).transient_local());
    gauss_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      gaussian_topic_, rclcpp::QoS(1).transient_local());
    voxel_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      voxel_topic_, rclcpp::QoS(1).transient_local());

    load_and_filter_once();
    build_map_once();

    // Publish periodically (1 Hz)
    timer_ = this->create_wall_timer(1s, [this]() { publish_once(); });

    RCLCPP_INFO(get_logger(),
                "Ready: pcd='%s' res=%.6f update_thresh=%d (publishing 1 Hz)",
                pcd_path_.c_str(), resolution_, update_threshold_);
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

  void build_map_once() {
    if (!filtered_cloud_ || filtered_cloud_->empty())
      throw std::runtime_error("Filtered cloud is empty");

    ivox_ = std::make_unique<gauss_ivox_mapping::GaussianIVox>(
      static_cast<gauss_ivox_mapping::Scalar>(resolution_),
      static_cast<std::size_t>(update_threshold_)
    );

    auto tick = std::chrono::system_clock::now(); 
    for (const auto& p : filtered_cloud_->points) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
      gauss_ivox_mapping::Point point;
      point << static_cast<gauss_ivox_mapping::Scalar>(p.x),
                  static_cast<gauss_ivox_mapping::Scalar>(p.y),
                  static_cast<gauss_ivox_mapping::Scalar>(p.z);
      ivox_->update(point);
    }
    auto tack = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_time = tack-tick;
    
    map_loaded_ = true;

    cached_gauss_.clear();
    cached_gauss_ = ivox_->getGaussians();

    RCLCPP_INFO(get_logger(), "Map contains %zu gaussians and took %f ms to load",
                cached_gauss_.size(), elapsed_time.count()*1000.0);
  }

  void publish_once() {
    if (!cloud_loaded_ || !map_loaded_) return;
    const auto stamp = this->now();

    cloud_msg_.header.stamp = stamp;
    cloud_pub_->publish(cloud_msg_);

    // gauss_pub_->publish(makeGaussianMarkers(stamp));
    gauss_pub_->publish(makeGaussianMarkers(*ivox_.get(), stamp));

    voxel_pub_->publish(makeVoxelMarkers(*ivox_.get(), stamp, resolution_));
  }

  visualization_msgs::msg::MarkerArray makeGaussianMarkers(
      const rclcpp::Time& stamp) const
  {
      visualization_msgs::msg::MarkerArray arr;
      if (!map_loaded_) return arr;

      visualization_msgs::msg::Marker del;
      del.header.frame_id = frame_id_;
      del.header.stamp = stamp;
      del.ns = "gaussians";
      del.id = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);

      const size_t n = cached_gauss_.size();
      arr.markers.reserve(n);

      int id = 0;

      for (size_t i = 0; i < n; ++i) {
          const auto& g = *cached_gauss_[i];

          visualization_msgs::msg::Marker m;

          std::cout << "Gauss Type: " << static_cast<int>(g.type) << " Count: " << g.count << std::endl; // --- DEBUG ---

          switch (g.type) {
              case gauss_ivox_mapping::PrimitiveType::PLANE:
                  m = makePlaneMarker(g, id++, stamp);
                  break;

              case gauss_ivox_mapping::PrimitiveType::VOLUME:
                  m = makeVolumeMarker(g, id++, stamp);
                  break;

              default:
                  continue;
          }

          arr.markers.push_back(m);
      }

      return arr;
  }

  visualization_msgs::msg::Marker makePlaneMarker(
      const gauss_ivox_mapping::GaussianPrimitive& g,
      int id,
      const rclcpp::Time& stamp) const
  {
      using Point = gauss_ivox_mapping::Point;

      visualization_msgs::msg::Marker m;

      const double plane_size = 1.0;
      const double thickness  = 0.02;

      auto buildOmega = [](const Point& p, int main_dir) {
        Point w;
        switch (main_dir) {
            case 0: w << p[0], p[1], 1.0; break;
            case 1: w << p[0], 1.0, p[1]; break;
            case 2: w << 1.0, p[0], p[1]; break;
        }
        return w;
      };

      Eigen::Vector3d normal = buildOmega(g.param, g.main_dir);
      double norm2 = normal.squaredNorm();
      Point p0 = g.mean;  // or voxel_center for grid alignment

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

      float k = 10.0f;
      float v = float(g.count) / (float(g.count) + k);

      m.color.r = v;
      m.color.g = 1.0f - std::fabs(v - 0.5f) * 2.0f;
      m.color.b = 1.0f - v;
      m.color.a = 0.8f;

      m.lifetime = rclcpp::Duration(0, 0);

      return m;
  }

  visualization_msgs::msg::Marker makeVolumeMarker(
      const gauss_ivox_mapping::GaussianPrimitive& g,
      int id,
      const rclcpp::Time& stamp) const
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

      float k = 10.0f;
      float v = float(g.count) / (float(g.count) + k);

      m.color.r = v;
      m.color.g = 1.0f - std::fabs(v - 0.5f) * 2.0f;
      m.color.b = 1.0f - v;
      m.color.a = 0.7f;

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

          // --- Voxel cube ---
          visualization_msgs::msg::Marker m;
          m.header.frame_id = "map";
          m.header.stamp = stamp;
          m.ns = "voxels";
          m.id = id++;
          m.type = visualization_msgs::msg::Marker::CUBE;
          m.action = visualization_msgs::msg::Marker::ADD;
          m.pose.position.x = node->voxel_center[0];
          m.pose.position.y = node->voxel_center[1];
          m.pose.position.z = node->voxel_center[2];
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

visualization_msgs::msg::MarkerArray makeGaussianMarkers(
      const gauss_ivox_mapping::GaussianIVox& ivox,
      const rclcpp::Time& stamp)
  {
      visualization_msgs::msg::MarkerArray arr;
      if (!map_loaded_) return arr;

      visualization_msgs::msg::Marker del;
      del.header.frame_id = frame_id_;
      del.header.stamp = stamp;
      del.ns = "gaussians";
      del.id = 0;
      del.action = visualization_msgs::msg::Marker::DELETEALL;
      arr.markers.push_back(del);
    //   std::shared_lock lock(ivox.map_mtx_); // safe map access

      std::mt19937 gen(12345); // fixed seed for consistent cluster colors
      std::uniform_real_distribution<float> dis(0.0f, 1.0f);
      std::unordered_map<gauss_ivox_mapping::UnionFindNode*, std::tuple<float,float,float>> parent_colors;

      int id = 0;
      for (const auto& [key, node] : ivox.map_) {
          auto* root = node->find();
          if (parent_colors.find(root) == parent_colors.end()) {
              parent_colors[root] = {dis(gen), dis(gen), dis(gen)};
          }

          auto& gaus = node->gauss_ptr;
          if (!gaus) continue; // if not root skip (will not have geometry)

        //   std::cout << "Gauss Type: " << static_cast<int>(gaus->type) << " Count: " << gaus->count << std::endl; // --- DEBUG ---

          visualization_msgs::msg::Marker m;
          id++; // increment here to ensure unique IDs even if type is unknown

          switch (gaus->type) {
              case gauss_ivox_mapping::PrimitiveType::PLANE:
                  std::cout << "Making plane marker for node with count: " << gaus->count << std::endl; // --- DEBUG ---
                  m = makePlaneMarker(*gaus, id, stamp, parent_colors[root]);
                  break;

              case gauss_ivox_mapping::PrimitiveType::VOLUME:
                  std::cout << "Making volume marker for node with count: " << gaus->count << std::endl; // --- DEBUG ---
                  m = makeVolumeMarker(*gaus, id, stamp, parent_colors[root]);
                  break;

              default:
                  break;
          }

          arr.markers.push_back(m);
      }
      
      return arr;
  }

  visualization_msgs::msg::Marker makePlaneMarker(
      const gauss_ivox_mapping::GaussianPrimitive& g,
      int id,
      const rclcpp::Time& stamp,
      std::tuple<float,float,float>& color) const
  {
      using Point = gauss_ivox_mapping::Point;

      visualization_msgs::msg::Marker m;

      const double plane_size = 2.0;
      const double thickness = 0.02;

      auto buildOmega = [](const Point& p, int main_dir) {
        Point w;
        switch (main_dir) {
            case 0: w << p[0], p[1], 1.0; break;
            case 1: w << p[0], 1.0, p[1]; break;
            case 2: w << 1.0, p[0], p[1]; break;
        }
        return w;
      };

      Eigen::Vector3d normal = buildOmega(g.param, g.main_dir);
      double norm2 = normal.squaredNorm();
      Point p0 = g.mean;  // or voxel_center for grid alignment

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


      m.scale.x = plane_size * sqrt(g.cov(0,0)); // spread along x-axis;
      m.scale.y = plane_size * sqrt(g.cov(1,1)); // spread along y-axis;
      m.scale.z = thickness;
    //   m.scale.z = sqrt(g.cov(2,2)); // spread along z-axis

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


private:
  // params
  std::string pcd_path_;
  double resolution_{5.0};
  int update_threshold_{5};
  double filter_x_{10.0};

  // fixed
  std::string frame_id_;
  std::string topic_;
  std::string gaussian_topic_;
  std::string voxel_topic_;

  // state
  bool cloud_loaded_{false};
  bool map_loaded_{false};
  sensor_msgs::msg::PointCloud2 cloud_msg_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr filtered_cloud_{nullptr};

  std::unique_ptr<gauss_ivox_mapping::GaussianIVox> ivox_{nullptr};
  std::vector<gauss_ivox_mapping::GaussPtr> cached_gauss_;

  // ros
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr gauss_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxel_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<GaussianIVOXStaticLoad>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("gaussian_ivox_test"), "Exception: %s", e.what());
  }

  rclcpp::shutdown();
  return 0;
}