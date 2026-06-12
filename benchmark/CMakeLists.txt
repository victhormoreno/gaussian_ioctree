cmake_minimum_required(VERSION 3.16)
project(map_structure_benchmarks CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Force aggressive mathematical optimizations and explicit hardware acceleration loops
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -DNDEBUG")

# Default to Release if not specified 
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Eigen3 REQUIRED)

find_package( PCL 1.12 REQUIRED )
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

message("PCL version: ${PCL_VERSION}")

# Use absolute paths relative to this CMake list to point exactly to your source include directory
include_directories(
    ${EIGEN3_INCLUDE_DIRS}
    ${PCL_INLCUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

add_executable(run_map_benchmark main.cpp)
target_link_libraries(run_map_benchmark Eigen3::Eigen ${PCL_LIBRARIES})