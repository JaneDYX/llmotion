from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='planner_manager',
            executable='planner_manager_node',
            name='planner_manager_node',
            output='screen',
            parameters=[]
        )
    ])
