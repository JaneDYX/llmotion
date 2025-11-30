cmake_minimum_required(VERSION 3.5)
project(planner_manager)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(carla_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
include_directories(
  include
)
add_executable(planner_manager_node
  src/planner_manager_node.cpp
  src/planner_lattice.cpp
  src/speed_profiler.cpp
)
ament_target_dependencies(planner_manager_node
  rclcpp nav_msgs geometry_msgs visualization_msgs tf2 tf2_ros std_msgs)

install(TARGETS planner_manager_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()

add_executable(trajectory_controller_node
  src/trajectory_controller_node.cpp
)

ament_target_dependencies(trajectory_controller_node
  rclcpp
  nav_msgs
  geometry_msgs
  carla_msgs
)

install(TARGETS
  trajectory_controller_node
  DESTINATION lib/${PROJECT_NAME}
)
