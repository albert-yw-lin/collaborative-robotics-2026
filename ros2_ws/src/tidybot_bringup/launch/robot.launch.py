"""
Full Robot Stack Launch File for TidyBot2.

This is the primary launch file for deploying TidyBot2 for remote control.
It launches all hardware drivers and configures the system for network access.

Features:
- Phoenix 6 mobile base with odometry and TF
- Dual Interbotix WX200 arm control
- Pan-tilt camera controller
- RealSense camera with optional compression
- Network-ready DDS configuration
- Motion planner (optional)

Usage:
    # Full robot stack (recommended for deployment)
    ros2 launch tidybot_bringup robot.launch.py

    # With image compression for remote clients
    ros2 launch tidybot_bringup robot.launch.py use_compression:=true

    # Without motion planner (lighter weight)
    ros2 launch tidybot_bringup robot.launch.py use_planner:=false

    # Headless mode (no RViz, for deployment)
    ros2 launch tidybot_bringup robot.launch.py use_rviz:=false

Network Configuration:
    Before launching, set up the DDS environment:

    # Option 1: Cyclone DDS with multicast (simple, same subnet)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file://$(ros2 pkg prefix tidybot_bringup)/share/tidybot_bringup/config/cyclone_dds_robot.xml
    export ROS_DOMAIN_ID=42

    # Option 2: FastDDS with discovery server (works across subnets)
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix tidybot_bringup)/share/tidybot_bringup/config/fastdds_robot.xml
    fastdds discovery --server-id 0 --port 11811 &
    export ROS_DOMAIN_ID=42

Remote Clients:
    Clients can connect by:
    1. Installing ROS2 Humble and tidybot_msgs package
    2. Setting matching ROS_DOMAIN_ID and DDS configuration
    3. Running: ros2 topic list (should see robot topics)
    4. Using standard ROS2 commands: ros2 topic pub /cmd_vel ...
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution, EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Package paths
    pkg_bringup = FindPackageShare('tidybot_bringup')
    pkg_description = FindPackageShare('tidybot_description')
    pkg_control = FindPackageShare('tidybot_control')

    # Declare arguments
    declare_use_rviz = DeclareLaunchArgument(
        'use_rviz', default_value='false',  # Default off for deployment
        description='Launch RViz for visualization'
    )

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation time (false for real robot)'
    )

    declare_use_base = DeclareLaunchArgument(
        'use_base', default_value='true',
        description='Launch Phoenix 6 mobile base driver'
    )

    declare_use_arms = DeclareLaunchArgument(
        'use_arms', default_value='true',
        description='Launch arm hardware drivers'
    )

    declare_use_pan_tilt = DeclareLaunchArgument(
        'use_pan_tilt', default_value='true',
        description='Launch pan-tilt camera controller'
    )

    declare_use_camera = DeclareLaunchArgument(
        'use_camera', default_value='true',
        description='Launch RealSense camera driver'
    )

    declare_use_compression = DeclareLaunchArgument(
        'use_compression', default_value='true',  # Default on for remote
        description='Launch image compression for remote clients'
    )

    declare_use_planner = DeclareLaunchArgument(
        'use_planner', default_value='true',
        description='Launch motion planner node'
    )

    declare_domain_id = DeclareLaunchArgument(
        'domain_id', default_value='42',
        description='ROS domain ID for network discovery'
    )

    # Set ROS_DOMAIN_ID if specified
    set_domain_id = SetEnvironmentVariable(
        name='ROS_DOMAIN_ID',
        value=LaunchConfiguration('domain_id')
    )

    # URDF from xacro
    urdf_path = PathJoinSubstitution([pkg_description, 'urdf', 'tidybot_wx200.urdf.xacro'])
    robot_description = Command(['xacro ', urdf_path])

    # ========== CORE NODES ==========

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }]
    )

    # ========== HARDWARE DRIVERS ==========

    # Phoenix 6 mobile base driver
    phoenix6_base = Node(
        package='tidybot_control',
        executable='phoenix6_base_node',
        name='phoenix6_base',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_base')),
        parameters=[{
            'max_linear_vel': 0.5,
            'max_linear_vel_y': 0.5,
            'max_angular_vel': 1.57,
            'max_linear_accel': 0.25,
            'max_angular_accel': 0.79,
            'publish_rate': 50.0,
            'position_tolerance': 0.02,
            'orientation_tolerance': 0.05,
        }]
    )

    # Right arm hardware interface
    right_arm_hw = Node(
        package='tidybot_control',
        executable='interbotix_arm_node',
        name='right_arm_hw',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arms')),
        parameters=[{
            'arm_name': 'right',
            'robot_model': 'wx250s',
            'publish_rate': 100.0,
        }]
    )

    # Left arm hardware interface
    left_arm_hw = Node(
        package='tidybot_control',
        executable='interbotix_arm_node',
        name='left_arm_hw',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arms')),
        parameters=[{
            'arm_name': 'left',
            'robot_model': 'wx250s',
            'publish_rate': 100.0,
        }]
    )

    # Pan-tilt camera controller
    pan_tilt = Node(
        package='tidybot_control',
        executable='pan_tilt_node',
        name='pan_tilt',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_pan_tilt')),
        parameters=[{
            'port_name': '/dev/ttyUSB0',
            'baudrate': 1000000,
            'pan_motor_id': 21,
            'tilt_motor_id': 22,
            'publish_rate': 50.0,
        }]
    )

    # RealSense camera
    realsense_camera = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_camera')),
        parameters=[{
            'camera_name': 'camera',
            'camera_namespace': '',
            'base_frame_id': 'link',
            'enable_color': True,
            'enable_depth': True,
            'enable_infra1': False,
            'enable_infra2': False,
            'publish_tf': False,
            'rgb_camera.color_profile': '640x480x15',
            'depth_module.depth_profile': '640x480x15',
        }],
        remappings=[
            ('/camera/realsense/color/image_raw', '/camera/color/image_raw'),
            ('/camera/realsense/depth/image_rect_raw', '/camera/depth/image_raw'),
            ('/camera/realsense/color/camera_info', '/camera/color/camera_info'),
            ('/camera/realsense/depth/camera_info', '/camera/depth/camera_info'),
        ]
    )

    # ========== CONTROLLERS ==========

    # Right arm controller
    right_arm_controller = Node(
        package='tidybot_control',
        executable='arm_controller_node',
        name='right_arm_controller',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arms')),
        parameters=[{
            'arm_name': 'right',
            'control_rate': 50.0,
        }]
    )

    # Left arm controller
    left_arm_controller = Node(
        package='tidybot_control',
        executable='arm_controller_node',
        name='left_arm_controller',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_arms')),
        parameters=[{
            'arm_name': 'left',
            'control_rate': 50.0,
        }]
    )

    # Motion planner
    motion_planner = Node(
        package='tidybot_ik',
        executable='motion_planner_node',
        name='motion_planner',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_planner')),
        parameters=[{
            'planning_rate': 10.0,
        }]
    )

    # ========== NETWORK BRIDGE ==========

    # Image compression for remote clients
    image_compression = Node(
        package='tidybot_network_bridge',
        executable='image_compression_node',
        name='image_compression',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_compression')),
        parameters=[{
            'jpeg_quality': 80,
            'png_level': 3,
            'target_fps': 15.0,
        }]
    )

    # ========== VISUALIZATION ==========

    # RViz (optional, off by default for deployment)
    rviz_config = PathJoinSubstitution([pkg_bringup, 'rviz', 'tidybot.rviz'])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        # Arguments
        declare_use_rviz,
        declare_use_sim_time,
        declare_use_base,
        declare_use_arms,
        declare_use_pan_tilt,
        declare_use_camera,
        declare_use_compression,
        declare_use_planner,
        declare_domain_id,

        # Environment
        set_domain_id,

        # Core
        robot_state_publisher,

        # Hardware drivers
        phoenix6_base,
        right_arm_hw,
        left_arm_hw,
        pan_tilt,
        realsense_camera,

        # Controllers
        right_arm_controller,
        left_arm_controller,
        motion_planner,

        # Network bridge
        image_compression,

        # Visualization
        rviz,
    ])
