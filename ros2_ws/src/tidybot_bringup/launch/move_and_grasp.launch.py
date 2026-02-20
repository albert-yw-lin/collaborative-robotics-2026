from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. Motion Planner -> 패키지를 'tidybot_ik'로 지정!
        Node(
            package='tidybot_ik',       # <--- 핵심 변경 사항
            executable='motion_planner_node', # setup.py에서 정한 이름
            name='motion_planner',
            output='screen'
        ),
        
        # 2. Combined Controller -> 패키지를 'tidybot_bringup'으로 유지
        Node(
            package='tidybot_bringup',
            executable='combined_arm_gripper_controller',
            name='combined_arm_gripper_controller',
            output='screen'
        )
    ])