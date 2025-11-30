#ifndef PLANNER_MANAGER_NODE_HPP_
#define PLANNER_MANAGER_NODE_HPP_

#include <memory>
#include <mutex>
#include <vector>

#include <visualization_msgs/msg/marker_array.hpp>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"

class PlannerManagerNode : public rclcpp::Node
{
public:
  PlannerManagerNode();

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
  void referencePathCallback(const nav_msgs::msg::Path::SharedPtr msg);
  void timerCallback();

  // --- Subscribers ---
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr reference_sub_;

  // --- Publishers ---
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectory_pub_;

  // --- Timer ---
  rclcpp::TimerBase::SharedPtr timer_;

  // --- Data ---
  std::mutex data_mutex_;
  nav_msgs::msg::Odometry last_odom_;
  nav_msgs::msg::Path reference_path_;
  bool has_odom_ = false;
  bool has_reference_path_ = false;

  // --- Trajectory Stitching Data ---
  nav_msgs::msg::Path last_trajectory_;
  bool has_last_traj_ = false;
};

#endif
