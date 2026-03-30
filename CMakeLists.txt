cmake_minimum_required(VERSION 3.8)
project(gaussian_octree)

# Silence dev warning about <PackageName>_ROOT variables (e.g., PCL_ROOT)
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

# Default to Release if not specified 
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(ENABLE_PROFILING "Enable debug RAII profiling" OFF)

# -----------------------
# Dependencies
# -----------------------
find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# For static_load_test (PCD loader + ROS2 publisher)
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

if(ENABLE_PROFILING)
target_compile_definitions(${PROJECT_NAME} INTERFACE ENABLE_PROFILING)
endif()

# -----------------------
# Test executable: load + publish PCD
# -----------------------
add_executable(static_load_test src/static_load_test.cpp)

ament_target_dependencies(static_load_test
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(static_load_test PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(static_load_test
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)
# OpenMP
if(OpenMP_CXX_FOUND)
  target_link_libraries(static_load_test OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP enabled for static_load_test")
else()
  message(WARNING "OpenMP not found: omp pragmas will be ignored")
endif()

# -----------------------
# Test executable: load + publish PCD with ivox
# -----------------------
add_executable(static_load_test_unionfind src/static_load_test_unionfind.cpp)

ament_target_dependencies(static_load_test_unionfind
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(static_load_test_unionfind PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(static_load_test_unionfind
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# -----------------------
# Test executable: real time incremental mapping test
# -----------------------
add_executable(real_time_test src/real_time_test.cpp)

ament_target_dependencies(real_time_test
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(real_time_test PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(real_time_test
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# -----------------------
# Test executable: real time incremental mapping test with ivox
# -----------------------
add_executable(real_time_test_ivox src/real_time_test_ivox.cpp)

ament_target_dependencies(real_time_test_ivox
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(real_time_test_ivox PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(real_time_test_ivox
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# -----------------------
# Test executable: incremental mapping test
# -----------------------
add_executable(incremental_mapping_test src/incremental_mapping_test.cpp)

ament_target_dependencies(incremental_mapping_test
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(incremental_mapping_test PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(incremental_mapping_test
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# -----------------------
# Test executable: incremental mapping test
# -----------------------
add_executable(real_time_test_unionfind src/real_time_test_unionfind.cpp)

ament_target_dependencies(real_time_test_unionfind
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  pcl_ros
)

target_include_directories(real_time_test_unionfind PRIVATE
  ${PCL_INCLUDE_DIRS}
)

# Link libraries 
target_link_libraries(real_time_test_unionfind
  ${PROJECT_NAME}
  ${PCL_LIBRARIES}
)

# -----------------------
# Install
# -----------------------
install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}
  static_load_test
  static_load_test_unionfind
  real_time_test
  real_time_test_ivox
  incremental_mapping_test
  real_time_test_unionfind
  EXPORT export_${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3)

ament_package()