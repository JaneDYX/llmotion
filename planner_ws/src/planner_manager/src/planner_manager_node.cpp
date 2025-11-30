#include "planner_manager/planner_manager_node.hpp"

#include <cmath>

using std::placeholders::_1;

PlannerManagerNode::PlannerManagerNode()
: rclcpp::Node("planner_manager_node")
{
  RCLCPP_INFO(this->get_logger(), "PlannerManagerNode started.");

  odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
    "/carla/hero/odometry", 10,
    std::bind(&PlannerManagerNode::odomCallback, this, _1));

  reference_sub_ = this->create_subscription<nav_msgs::msg::Path>(
    "/map/reference_path", 10,
    std::bind(&PlannerManagerNode::referencePathCallback, this, _1));

  trajectory_pub_ = this->create_publisher<nav_msgs::msg::Path>(
    "/planning/trajectory", 10);

  timer_ = this->create_wall_timer(
    std::chrono::milliseconds(100),
    std::bind(&PlannerManagerNode::timerCallback, this));
}

void PlannerManagerNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  last_odom_ = *msg;
  has_odom_ = true;
}

void PlannerManagerNode::referencePathCallback(const nav_msgs::msg::Path::SharedPtr msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  reference_path_ = *msg;
  has_reference_path_ = true;
  RCLCPP_INFO(this->get_logger(), "Received reference path with %zu poses", reference_path_.poses.size());
}

void PlannerManagerNode::timerCallback()
{
  std::lock_guard<std::mutex> lock(data_mutex_);

  if (!has_odom_ || !has_reference_path_ || reference_path_.poses.empty()) {
    return;
  }

  nav_msgs::msg::Path new_traj_raw;
  new_traj_raw.header.stamp = this->now();
  new_traj_raw.header.frame_id = "map";

  // -----------------------------
  // 1. 生成 new trajectory（你的原始逻辑）
  // -----------------------------
  const auto & ref = reference_path_.poses;
  double px = last_odom_.pose.pose.position.x;
  double py = last_odom_.pose.pose.position.y;

  size_t closest_idx = 0u;
  double min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < ref.size(); ++i) {
    double dx = ref[i].pose.position.x - px;
    double dy = ref[i].pose.position.y - py;
    double d = std::hypot(dx, dy);
    if (d < min_dist) {
      min_dist = d;
      closest_idx = i;
    }
  }

  size_t horizon = 40u;
  for (size_t i = closest_idx; i < ref.size() && i < closest_idx + horizon; ++i) {
    new_traj_raw.poses.push_back(ref[i]);
  }

  if (new_traj_raw.poses.empty()) return;

  // ---------------------------------------------------
  // 2. 若没有 old trajectory，直接输出 new trajectory
  // ---------------------------------------------------
  if (!has_last_traj_) {
    trajectory_pub_->publish(new_traj_raw);
    last_trajectory_ = new_traj_raw;
    has_last_traj_ = true;
    return;
  }

  // ---------------------------------------------------
  // 3. 找到 old trajectory 中与车辆当前最近的点
  // ---------------------------------------------------
  const auto & old_traj = last_trajectory_.poses;

  size_t old_closest_idx = 0u;
  double old_min_dist = std::numeric_limits<double>::max();
  for (size_t i = 0; i < old_traj.size(); ++i) {
    double dx = old_traj[i].pose.position.x - px;
    double dy = old_traj[i].pose.position.y - py;
    double d = std::hypot(dx, dy);
    if (d < old_min_dist) {
      old_min_dist = d;
      old_closest_idx = i;
    }
  }

  // ---------------------------------------------------
  // 4. 取 old trajectory 的末尾 tail（平滑过渡段）
  // ---------------------------------------------------
  size_t tail_keep = 10;  // 通常取 5–20 比较合适
  nav_msgs::msg::Path stitched_traj;
  stitched_traj.header = new_traj_raw.header;

  for (size_t i = old_closest_idx;
       i < old_traj.size() && stitched_traj.poses.size() < tail_keep;
       ++i)
  {
    stitched_traj.poses.push_back(old_traj[i]);
  }


  // ---------------------------------------------------
  // 6. 拼接：old_tail + aligned_new_traj
  // ---------------------------------------------------
  for (const auto & p : new_traj_raw.poses) {
    stitched_traj.poses.push_back(p);
  }

  // ---------------------------------------------------
  // 7. 发布 + 保存为 last trajectory
  // ---------------------------------------------------
  trajectory_pub_->publish(stitched_traj);
  last_trajectory_ = stitched_traj;
  has_last_traj_ = true;
}


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PlannerManagerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
