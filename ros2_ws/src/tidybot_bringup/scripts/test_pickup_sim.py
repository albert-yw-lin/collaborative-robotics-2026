#!/usr/bin/env python3
"""
TidyBot2 Block Pickup with Grasp Detection (ROS2 Simulation)

Picks up a block using the right arm with IK planning and verifies
the grasp using GripperController.check_grasp().

Usage:
    # Terminal 1: Start simulation with the pickup scene
    ros2 launch tidybot_bringup sim.launch.py scene:=scene_pickup.xml

    # Terminal 2: Run this test
    ros2 run tidybot_bringup test_pickup_sim.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from tidybot_msgs.srv import PlanToTarget
from tidybot_control.gripper_controller import GripperController
import time


# Block position from scene_pickup.xml (world frame = base_link when base is at origin)
BLOCK_POS = (0.55, -0.15, 0.025)

# Offsets above the block centre
APPROACH_HEIGHT = 0.15   # 15 cm above block
GRASP_HEIGHT = 0.06      # fingertips at block top
LIFT_HEIGHT = 0.20       # lift to 20 cm above block

# Top-down grasp orientation (fingers pointing down) in base_link frame
# quaternion (qw, qx, qy, qz)
ORIENT_FINGERS_DOWN = (0.5, 0.5, 0.5, -0.5)


class TestPickup(Node):

    def __init__(self):
        super().__init__('test_pickup_sim')

        # IK planning service client
        self.plan_client = self.create_client(PlanToTarget, '/plan_to_target')

        # Gripper controller (sim mode)
        self.gripper = GripperController(self, mode='sim')

        self.get_logger().info('Waiting for /plan_to_target service...')
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  Service not available, waiting...')
        self.get_logger().info('Service connected!')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def make_pose(self, x, y, z):
        """Create a Pose with fingers-down orientation."""
        qw, qx, qy, qz = ORIENT_FINGERS_DOWN
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.w = qw
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        return pose

    def plan_and_execute(self, pose, duration=2.0):
        """Send a PlanToTarget request and block until done."""
        req = PlanToTarget.Request()
        req.arm_name = 'right'
        req.target_pose = pose
        req.use_orientation = True
        req.execute = True
        req.duration = duration
        req.max_condition_number = 100.0

        future = self.plan_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=15.0)

        if not future.done() or future.exception():
            self.get_logger().error('Service call failed!')
            return False

        result = future.result()
        if result.success:
            self.get_logger().info(f'  OK — pos_err={result.position_error:.4f}m')
            time.sleep(duration + 0.5)  # wait for motion to finish
            return True
        else:
            self.get_logger().warn(f'  FAILED: {result.message}')
            return False

    # ------------------------------------------------------------------
    # Main pickup sequence
    # ------------------------------------------------------------------
    def run(self):
        bx, by, bz = BLOCK_POS
        self.get_logger().info(f'Block at ({bx}, {by}, {bz})')

        # 1 — Open gripper
        self.get_logger().info('[1/5] Opening gripper...')
        self.gripper.open('right', duration=1.5)

        # 2 — Approach (above the block)
        self.get_logger().info('[2/5] Moving to approach position...')
        approach = self.make_pose(bx, by, bz + APPROACH_HEIGHT)
        if not self.plan_and_execute(approach, duration=2.5):
            self.get_logger().error('Could not reach approach pose — aborting.')
            return

        # 3 — Descend to grasp height
        self.get_logger().info('[3/5] Descending to grasp position...')
        grasp = self.make_pose(bx, by, bz + GRASP_HEIGHT)
        if not self.plan_and_execute(grasp, duration=2.0):
            self.get_logger().error('Could not reach grasp pose — aborting.')
            return

        # 4 — Close gripper + check grasp
        self.get_logger().info('[4/5] Closing gripper...')
        grasped = self.gripper.close_and_check('right', duration=2.0)
        if grasped:
            self.get_logger().info('  Grasp detected!')
        else:
            self.get_logger().warn('  No object detected between fingers.')

        # 5 — Lift
        self.get_logger().info('[5/5] Lifting...')
        lift = self.make_pose(bx, by, bz + LIFT_HEIGHT)
        self.plan_and_execute(lift, duration=2.0)

        self.get_logger().info('')
        self.get_logger().info('=' * 40)
        self.get_logger().info(f'Pickup complete — grasped={grasped}')
        self.get_logger().info('=' * 40)


def main(args=None):
    rclpy.init(args=args)
    node = TestPickup()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
