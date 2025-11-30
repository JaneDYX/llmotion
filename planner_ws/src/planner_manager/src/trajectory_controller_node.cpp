#include <memory>
#include <vector>
#include <cmath>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "carla_msgs/msg/carla_ego_vehicle_control.hpp"

class TrajectoryControllerNode : public rclcpp::Node
{
public:
  TrajectoryControllerNode()
  : Node("trajectory_controller_node"),
    wheelbase_(2.8),
    lookahead_distance_(5.0),
    target_speed_(8.0),
    max_steer_rad_(0.7),
    has_odom_(false),
    stuck_counter_(0),
    recovering_(false)
  {
    traj_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "/planning/trajectory", 10,
      std::bind(&TrajectoryControllerNode::trajCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/carla/hero/odometry", 10,
      std::bind(&TrajectoryControllerNode::odomCallback, this, std::placeholders::_1));

    control_pub_ = this->create_publisher<carla_msgs::msg::CarlaEgoVehicleControl>(
      "/carla/hero/vehicle_control_cmd", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50),
      std::bind(&TrajectoryControllerNode::timerCallback, this));

    RCLCPP_INFO(this->get_logger(), "TrajectoryControllerNode started with auto recovery.");
  }

private:

  // -----------------------------
  // Auto Unstuck Recovery Parameters
  // -----------------------------
  int stuck_counter_;
  bool recovering_;
  rclcpp::Time recover_end_time_;

  // Utility clamp
  static double clampValue(double value, double min_value, double max_value)
  {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
  }

  void trajCallback(const nav_msgs::msg::Path::SharedPtr msg)
  {
    latest_traj_ = *msg;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    latest_odom_ = *msg;
    has_odom_ = true;
  }

  void timerCallback()
  {
    if (!has_odom_ || latest_traj_.poses.empty()) {
      return;
    }

    // ---------------------------------
    // 1. Check vehicle stuck condition
    // ---------------------------------
    double vx = latest_odom_.twist.twist.linear.x;
    bool low_speed = std::fabs(vx) < 0.3;     // consider stuck if < 0.3 m/s
    bool trying_to_move = true;              // always true because planner wants to move

    if (!recovering_) {
      if (low_speed && trying_to_move) {
        stuck_counter_++;
      } else {
        stuck_counter_ = 0;
      }

      if (stuck_counter_ > 10) {  // ~0.5 seconds
        recovering_ = true;
        recover_end_time_ = this->now() + rclcpp::Duration::from_seconds(1.0);
        RCLCPP_WARN(this->get_logger(), "Vehicle stuck detected! Starting reverse recovery.");
      }
    }

    // ---------------------------------
    // 2. During recovery → reverse
    // ---------------------------------
    if (recovering_) {
      if (this->now() < recover_end_time_) {
        carla_msgs::msg::CarlaEgoVehicleControl rev;
        rev.throttle = 0.4;   // reverse speed
        rev.brake = 0.0;
        rev.steer = 0.0;
        rev.reverse = true;
        rev.manual_gear_shift = false;

        control_pub_->publish(rev);
        return;
      } else {
        recovering_ = false;
        stuck_counter_ = 0;
        RCLCPP_WARN(this->get_logger(), "Recovery completed — resuming normal tracking.");
      }
    }

    // ---------------------------------
    // 3. Normal Pure Pursuit control (your original tracking code)
    // ---------------------------------
    const auto & pose = latest_odom_.pose.pose;
    double x = pose.position.x;
    double y = pose.position.y;

    double qw = pose.orientation.w;
    double qx = pose.orientation.x;
    double qy = pose.orientation.y;
    double qz = pose.orientation.z;
    double siny_cosp = 2.0 * (qw * qz + qx * qy);
    double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    double yaw = std::atan2(siny_cosp, cosy_cosp);

    double min_dist = 1e9;
    int closest_idx = -1;
    for (size_t i = 0; i < latest_traj_.poses.size(); ++i) {
      double dx = latest_traj_.poses[i].pose.position.x - x;
      double dy = latest_traj_.poses[i].pose.position.y - y;
      double d = std::hypot(dx, dy);
      if (d < min_dist) {
        min_dist = d;
        closest_idx = static_cast<int>(i);
      }
    }

    if (closest_idx < 0) return;

    double accum_dist = 0.0;
    int lookahead_idx = closest_idx;
    for (size_t i = closest_idx + 1; i < latest_traj_.poses.size(); ++i) {
      double dx = latest_traj_.poses[i].pose.position.x -
                  latest_traj_.poses[i - 1].pose.position.x;
      double dy = latest_traj_.poses[i].pose.position.y -
                  latest_traj_.poses[i - 1].pose.position.y;
      accum_dist += std::hypot(dx, dy);
      lookahead_idx = static_cast<int>(i);
      if (accum_dist >= lookahead_distance_) {
        break;
      }
    }

    const auto & target_pose = latest_traj_.poses[lookahead_idx].pose;

    double dx = target_pose.position.x - x;
    double dy = target_pose.position.y - y;
    double x_local =  std::cos(-yaw) * dx - std::sin(-yaw) * dy;
    double y_local =  std::sin(-yaw) * dx + std::cos(-yaw) * dy;

    double ld = std::hypot(x_local, y_local);
    if (ld < 1e-3) return;

    double steering_angle = std::atan2(2.0 * wheelbase_ * y_local, ld * ld);
    double steer_norm = clampValue(steering_angle / max_steer_rad_, -1.0, 1.0);

    double speed_error = target_speed_ - vx;

    carla_msgs::msg::CarlaEgoVehicleControl cmd;
    cmd.steer = static_cast<float>(steer_norm);

    if (speed_error > 0.0) {
      cmd.throttle = static_cast<float>(clampValue(speed_error * 0.2, 0.0, 1.0));
      cmd.brake = 0.0f;
    } else {
      cmd.throttle = 0.0f;
      cmd.brake = static_cast<float>(clampValue(-speed_error * 0.2, 0.0, 1.0));
    }

    cmd.hand_brake = false;
    cmd.reverse = false;
    cmd.manual_gear_shift = false;

    control_pub_->publish(cmd);
  }

  // ----------------------------------
  // ROS member variables
  // ----------------------------------
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr traj_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<carla_msgs::msg::CarlaEgoVehicleControl>::SharedPtr control_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  nav_msgs::msg::Path latest_traj_;
  nav_msgs::msg::Odometry latest_odom_;
  bool has_odom_;

  double wheelbase_;
  double lookahead_distance_;
  double target_speed_;
  double max_steer_rad_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrajectoryControllerNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
