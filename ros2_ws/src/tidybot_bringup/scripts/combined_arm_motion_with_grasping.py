#!/usr/bin/env python3
"""
Combined Arm & Gripper Controller Node for TidyBot2.

This node acts as an integration layer to test the manipulation pipeline 
(motion planning + gripper control) independently from the terminal.
It uses separated callback groups and async service calls to guarantee 
no deadlocks occur during execution.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from tidybot_msgs.srv import PlanToTarget
from tidybot_control.gripper_controller import GripperController

class Combined_Arm_Gripper_Controller(Node):
    def __init__(self):
        super().__init__('combined_arm_gripper_controller')

        self.declare_parameter('gripper_mode', 'sim')
        gripper_mode = self.get_parameter('gripper_mode').get_parameter_value().string_value

        # Separate callback groups for Server and Client to avoid starvation
        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        # 1. Initialize Gripper Controller
        self.get_logger().info(f'[INIT] Initializing GripperController in {gripper_mode} mode...')
        self.gripper = GripperController(self, mode=gripper_mode, pressure=1.0)

        # 2. Client setup
        self.plan_client = self.create_client(
            PlanToTarget,
            '/plan_to_target',
            callback_group=self.client_cb_group
        )

        self.get_logger().info('[INIT] Waiting for /plan_to_target service...')
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('       Still waiting for /plan_to_target...')

        # 3. Server setup
        self.grasp_server = self.create_service(
            PlanToTarget,
            '/perform_grasp',
            self.grasp_callback,
            callback_group=self.server_cb_group
        )

        self.get_logger().info('[READY] Manipulation Coordinator ready. Call /perform_grasp to test.')

    def grasp_callback(self, request, response):
        arm_name = request.arm_name.lower()
        self.get_logger().info(f"\n========== GRASP SEQUENCE STARTED FOR {arm_name.upper()} ARM ==========")
        
        # Step 1: Open Gripper
        self.get_logger().info("STEP 1: Opening gripper...")
        self.gripper.open(arm_name, duration=1.0)
        
        # Step 2: Call Motion Planner (Async)
        self.get_logger().info("STEP 2: Calling motion planner...")
        future = self.plan_client.call_async(request)

        # Wait for the future safely
        wait_start = time.time()
        timeout_sec = 20.0
        
        while not future.done():
            if time.time() - wait_start > timeout_sec:
                self.get_logger().error(f"STEP 2 ERROR: Motion planner timeout!")
                response.success = False
                response.message = "Motion planner service timeout."
                return response
            time.sleep(0.05) # Yielding

        plan_response = future.result()

        if not plan_response.success:
            self.get_logger().error(f"STEP 2 ERROR: Planning failed -> {plan_response.message}")
            response.success = False
            response.message = f"Failed at motion planning: {plan_response.message}"
            return response

        self.get_logger().info("STEP 2: Motion planning successful!")

        # Step 3 & 4: Wait and Close Gripper
        if request.execute:
            self.get_logger().info(f"STEP 3: Waiting {request.duration}s for physical arm movement...")
            time.sleep(request.duration)

            self.get_logger().info("STEP 4: Target reached. Closing gripper...")
            self.gripper.close(arm_name, duration=2.0)
            response.message = "Grasp execution completed successfully."
        else:
            response.message = "Planning completed (execution skipped)."

        response.success = True
        response.position_error = plan_response.position_error
        response.orientation_error = plan_response.orientation_error
        response.joint_positions = plan_response.joint_positions
        
        self.get_logger().info(f"========== GRASP SEQUENCE COMPLETED ==========\n")
        return response

def main(args=None):
    rclpy.init(args=args)
    node = Combined_Arm_Gripper_Controller()
    
    executor = MultiThreadedExecutor(num_threads=4)
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