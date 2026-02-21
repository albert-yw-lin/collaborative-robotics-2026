#!/usr/bin/env python3
"""
TidyBot2 Block Pickup with Grasp Detection - Standalone MuJoCo Simulation

Based on pick_up_block_sim.py, adds grasp detection using the same logic as
tidybot_control.gripper_controller.GripperController.check_grasp().

After closing the gripper, checks whether the fingers stopped short of the
fully-closed position — meaning an object is held between them.

Usage:
    cd simulation/scripts
    uv run python test_pickup_grasp_sim.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

import mink


# Grasp detection constants (from gripper_controller.py)
FINGER_OPEN_POS = 0.037     # Fully open finger position (meters)
FINGER_CLOSED_POS = 0.015   # Fully closed finger position (meters)
GRASP_THRESHOLD = 0.003     # 3 mm above closed = object detected


def check_grasp(model, data, threshold=GRASP_THRESHOLD):
    """
    Check if the right gripper successfully grasped an object.

    Same logic as GripperController.check_grasp(): if the finger joints
    stopped above (FINGER_CLOSED_POS + threshold), something is between them.

    Returns:
        (grasped: bool, avg_pos: float)
    """
    left_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_left_finger")
    right_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_right_finger")

    left_pos = data.qpos[model.jnt_qposadr[left_jid]]
    right_pos = data.qpos[model.jnt_qposadr[right_jid]]
    avg_pos = (left_pos + right_pos) / 2.0

    grasped = avg_pos > (FINGER_CLOSED_POS + threshold)
    return grasped, avg_pos


def main():
    # ==========================================================================
    # Load Model
    # ==========================================================================
    script_dir = Path(__file__).parent
    model_path = script_dir / "../assets/mujoco/scene_pickup.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
    data = mujoco.MjData(model)

    # ==========================================================================
    # Get IDs
    # ==========================================================================
    block_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    right_arm_joints = [
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
    ]

    joint_qpos_addrs = {}
    for jname in right_arm_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        joint_qpos_addrs[jname] = model.jnt_qposadr[jid]

    right_arm_actuators = {}
    for jname in right_arm_joints:
        right_arm_actuators[jname] = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname
        )

    right_gripper_ctrl = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_fingers_actuator"
    )

    # ==========================================================================
    # Initialize
    # ==========================================================================
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    block_pos = data.xpos[block_body_id].copy()
    print(f"Block position: {block_pos}")

    # ==========================================================================
    # Setup Mink IK
    # ==========================================================================
    configuration = mink.Configuration(model)
    configuration.update(data.qpos)

    ee_task = mink.FrameTask(
        frame_name="right_pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    top_down_quat = np.array([0, 1, 0, 0])  # wxyz: 180 deg around X

    # ==========================================================================
    # Define waypoints
    # ==========================================================================
    approach_pos = block_pos + np.array([0, 0, 0.15])
    grasp_pos = block_pos + np.array([0, 0, 0.06])
    lift_pos = block_pos + np.array([0, 0, 0.20])

    print(f"Approach: {approach_pos}")
    print(f"Grasp:    {grasp_pos}")
    print(f"Lift:     {lift_pos}")

    # ==========================================================================
    # Run Simulation
    # ==========================================================================
    print("\nStarting pickup sequence...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.2
        viewer.cam.elevation = -30
        viewer.cam.azimuth = 150
        viewer.cam.lookat[:] = [0.4, -0.1, 0.2]
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY

        start_time = time.time()
        phase = "approach"
        grasp_checked = False

        dt = model.opt.timestep

        while viewer.is_running():
            step_start = time.time()
            elapsed = time.time() - start_time

            # ==================================================================
            # State Machine
            # ==================================================================
            if phase == "approach":
                target_pos = approach_pos
                if elapsed > 3.0:
                    phase = "descend"
                    print("Phase: descend")

            elif phase == "descend":
                t = min((elapsed - 3.0) / 2.0, 1.0)
                target_pos = approach_pos + t * (grasp_pos - approach_pos)
                if elapsed > 5.0:
                    phase = "grasp"
                    print("Phase: grasp — closing gripper")

            elif phase == "grasp":
                target_pos = grasp_pos
                gripper_progress = min((elapsed - 5.0) / 1.0, 1.0)
                data.ctrl[right_gripper_ctrl] = gripper_progress * 255

                if elapsed > 6.0 and not grasp_checked:
                    # --- Grasp detection (same logic as GripperController) ---
                    grasped, finger_pos = check_grasp(model, data)
                    print(f"\n  Grasp check: finger_pos={finger_pos:.4f}m, "
                          f"closed={FINGER_CLOSED_POS}m, "
                          f"threshold={GRASP_THRESHOLD}m, "
                          f"grasped={grasped}")
                    if grasped:
                        print("  Object detected between fingers!")
                    else:
                        print("  No object detected — fingers fully closed.")
                    grasp_checked = True

                    phase = "lift"
                    print("\nPhase: lift")

            elif phase == "lift":
                t = min((elapsed - 6.0) / 2.0, 1.0)
                target_pos = grasp_pos + t * (lift_pos - grasp_pos)
                data.ctrl[right_gripper_ctrl] = 255
                if elapsed > 8.0:
                    phase = "done"
                    print("Phase: done — pickup complete!")

            else:  # done
                target_pos = lift_pos
                data.ctrl[right_gripper_ctrl] = 255

            # ==================================================================
            # Solve IK
            # ==================================================================
            configuration.update(data.qpos)

            target_pose = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(top_down_quat),
                translation=target_pos,
            )
            ee_task.set_target(target_pose)

            vel = mink.solve_ik(
                configuration,
                [ee_task],
                dt=dt,
                solver="quadprog",
                damping=1e-3,
            )
            configuration.integrate_inplace(vel, dt)

            for jname in right_arm_joints:
                qpos_addr = joint_qpos_addrs[jname]
                act_id = right_arm_actuators[jname]
                data.ctrl[act_id] = configuration.q[qpos_addr]

            # ==================================================================
            # Step Simulation
            # ==================================================================
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed_step = time.time() - step_start
            sleep_time = dt - elapsed_step
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("Simulation ended.")


if __name__ == "__main__":
    main()
