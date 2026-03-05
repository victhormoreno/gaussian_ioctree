cmake_minimum_required(VERSION 3.8)
project(gaussian_octree)

# Silence dev warning about <PackageName>_ROOT variables (e.g., PCL_ROOT)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Default to Release if not specified )
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------
# Dependencies
# -----------------------
find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# For gaussian_octree_test (PCD loader + ROS2 publisher)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(PCL REQUIRED COMPONENTS io filters)

# OpenMP (optional)
find_package(OpenMP)

# -----------------------
# Header-only library
# -----------------------
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

# -----------------------
# Test executable: load + publish PCD
# -----------------------
add_executable(gaussian_octree_test src/gaussian_octree_test.cpp)

ament_target_dependencies(gaussian_octree_test
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(gaussian_octree_test PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(gaussian_octree_test
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(gaussian_octree_test OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled for gaussian_octree_test")
else()
  message(WARNING "OpenMP not found: omp pragmas will be ignored")
endif()

# -----------------------
# Install
# -----------------------
install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}
  gaussian_octree_test
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

ament_package()