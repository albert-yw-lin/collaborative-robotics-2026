#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Float64MultiArray  # Message type for gripper control
from tidybot_msgs.srv import PlanToTarget
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

# [Core] Removed external import to avoid deadlock.
# Instead, we define a safe gripper controller class directly here.
# from tidybot_control.gripper_controller import GripperController (Deleted)

class SafeGripperController:
    """
    A thread-safe Gripper Controller that does NOT use rclpy.spin_once().
    This prevents blocking the main executor of the node.
    """
    def __init__(self, node: Node):
        self.node = node
        # Publishers for simulation (Float64MultiArray)
        self.right_pub = node.create_publisher(Float64MultiArray, '/right_gripper/cmd', 10)
        self.left_pub = node.create_publisher(Float64MultiArray, '/left_gripper/cmd', 10)

    def _publish(self, side: str, position: float):
        """Helper to publish command to the correct topic."""
        msg = Float64MultiArray()
        msg.data = [float(position)]
        if side == 'right':
            self.right_pub.publish(msg)
        else:
            self.left_pub.publish(msg)

    def open(self, side: str):
        """
        Open the gripper.
        0.0 = Open (in simulation)
        Sends the command repeatedly for 1 second to ensure execution.
        """
        start = time.time()
        while time.time() - start < 1.0:
            self._publish(side, 0.0)
            time.sleep(0.05) # Just sleep, no spin_once to avoid conflicts.

    def close(self, side: str):
        """
        Close the gripper.
        1.0 = Closed
        Sends the command repeatedly for 1 second.
        """
        start = time.time()
        while time.time() - start < 1.0:
            self._publish(side, 1.0)
            time.sleep(0.05) # Just sleep, no spin_once.

class Combined_Arm_Gripper_Controller(Node):
    def __init__(self):
        super().__init__('combined_arm_gripper_controller')
        
        # Use ReentrantCallbackGroup to allow concurrent callbacks (prevent deadlock)
        self.callback_group = ReentrantCallbackGroup()

        # Use the safe internal class defined above
        self.gripper = SafeGripperController(self)
        
        # Client for the arm motion planner
        self.arm_motion_client = self.create_client(
            PlanToTarget, 
            '/plan_to_target',
            callback_group=self.callback_group
        )

        # Service server to receive commands from the terminal
        self.srv = self.create_service(
            PlanToTarget, 
            '/move_and_grasp', 
            self.handle_service, 
            callback_group=self.callback_group
        )

        self.get_logger().info('Waiting for plan_to_target service...')
        while not self.arm_motion_client.wait_for_service(timeout_sec=10):
            self.get_logger().info('Service not available, waiting again...')
        self.get_logger().info('plan_to_target service is available!')

    def handle_service(self, request, response):
        """
        Callback function for /move_and_grasp service.
        Executes: Open Gripper -> Move Arm -> Close Gripper.
        """
        self.get_logger().info(f'Received move_and_grasp request for arm: {request.arm_name}')

        # --- Step 1: Open Gripper ---
        self.get_logger().info('Step 1 : Opening gripper...')
        self.gripper.open(request.arm_name)
        # SafeGripperController already waits internally, so no extra sleep needed here.

        # --- Step 2: Move Arm ---
        self.get_logger().info('Step 2 : Sending Motion Request...')
        
        # Send asynchronous request to Motion Planner
        future = self.arm_motion_client.call_async(request)

        # Wait for the planning response (Non-blocking wait)
        while not future.done():
            time.sleep(0.01)
        
        motion_result = future.result()
        self.get_logger().info('   -> Motion Request Accepted by Planner!')

        # Wait for physical execution
        # The planner returns immediately after sending the command, so we must wait
        # for the duration of the movement before closing the gripper.
        if motion_result.success:
            wait_time = request.duration + 0.5 # Add 0.5s buffer
            self.get_logger().info(f'   -> Plan Succeeded! Waiting {wait_time:.1f}s for physical motion...')
            time.sleep(wait_time)

        # --- Step 3: Close Gripper ---
        if motion_result.success:
            self.get_logger().info('Step 3 : Now Grasping...')
            self.gripper.close(request.arm_name)
            
            response.success = True
            response.message = 'Arm moved and grasped successfully.'
            response.position_error = motion_result.position_error
            self.get_logger().info('Done!')
        else:
            self.get_logger().error(f'   -> Motion Failed: {motion_result.message}')
            response.success = False
            response.message = f"Motion failed: {motion_result.message}"    

        return response
    

def main(args=None):
    rclpy.init(args=args)
    node = Combined_Arm_Gripper_Controller()

    # MultiThreadedExecutor is essential for ReentrantCallbackGroup to work
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()