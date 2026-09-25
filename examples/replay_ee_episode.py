import argparse
import time

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.robots.so_follower.robot_kinematic_processor import (
    InverseKinematicsEEToJoints,
)
from lerobot.utils.constants import ACTION


def main():

    # ============================================================
    # Arguments
    # ============================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset.repo_id",
        type=str,
        required=True,
        dest="dataset_repo_id",
        help="HuggingFace LeRobot dataset",
    )

    parser.add_argument(
        "--dataset.episode",
        type=int,
        default=0,
        dest="dataset_episode",
        help="Episode to replay",
    )

    parser.add_argument(
        "--robot.port",
        type=str,
        required=True,
        dest="robot_port",
        help="SO101 follower serial port",
    )

    parser.add_argument(
        "--robot.id",
        type=str,
        default="my_awesome_follower_arm",
        dest="robot_id",
        help="Robot ID",
    )

    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to calibrated SO101 URDF",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Calculate IK but DO NOT move the robot",
    )

    args = parser.parse_args()

    # ============================================================
    # Dataset
    # ============================================================

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        episodes=[args.dataset_episode],
    )

    print()
    print("=" * 70)
    print("EE SPACE EPISODE REPLAY")
    print("=" * 70)

    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Episode: {args.dataset_episode}")
    print(f"FPS:     {dataset.fps}")
    print(f"Frames:  {dataset.num_frames}")

    print()
    print("Dataset action features:")
    print(dataset.features[ACTION]["names"])
    print()

    # ============================================================
    # Robot configuration
    # ============================================================

    robot_config = SO100FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        use_degrees=True,
    )

    robot = SO100Follower(robot_config)

    # ============================================================
    # Create LeRobot kinematics solver
    # ============================================================

    kinematics_solver = RobotKinematics(
        urdf_path=args.urdf,

        # This is the SO101 end-effector link
        target_frame_name="gripper_frame_link",

        # Use the robot's actual motor names
        joint_names=list(robot.bus.motors.keys()),
    )

    print("Robot joints:")
    print(list(robot.bus.motors.keys()))
    print()

    # ============================================================
    # Create the EE -> Joint IK processor
    # ============================================================

    ee_to_joint_processor = RobotProcessorPipeline(
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,

                motor_names=list(robot.bus.motors.keys()),

                # IMPORTANT:
                #
                # False = open-loop replay.
                #
                # The IK solution from the previous frame is
                # used as the initial guess for the next frame.
                #
                initial_guess_current_joints=False,
            ),
        ],

        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    print("IK processor created.")
    print()

    # ============================================================
    # Dry run
    # ============================================================

    if args.dry_run:

        print("=" * 70)
        print("DRY RUN")
        print("Robot will NOT move.")
        print("=" * 70)
        print()

    else:

        print("Connecting to robot...")
        robot.connect()

        if not robot.is_connected:
            raise RuntimeError("Could not connect to SO101")

        print("Robot connected.")
        print()

        print("Starting in 3 seconds...")
        time.sleep(3)

    # ============================================================
    # Replay episode
    # ============================================================

    try:

        for frame_idx in range(dataset.num_frames):

            frame_start = time.perf_counter()

            # ----------------------------------------------------
            # Get EE action from dataset
            # ----------------------------------------------------

            action_tensor = dataset[frame_idx][ACTION]

            action_names = dataset.features[ACTION]["names"]

            ee_action = {
                name: float(action_tensor[i])
                for i, name in enumerate(action_names)
            }

            # ----------------------------------------------------
            # Print recorded EE action
            # ----------------------------------------------------

            print()
            print("-" * 70)
            print(f"FRAME {frame_idx}")
            print("-" * 70)

            print("Recorded EE action:")

            for name, value in ee_action.items():
                print(f"  {name:15s}: {value:.6f}")

            # ----------------------------------------------------
            # Get robot observation
            # ----------------------------------------------------

            if args.dry_run:

                # We don't need a real observation for this
                # basic IK test.
                robot_obs = {}

            else:

                robot_obs = robot.get_observation()

            # ----------------------------------------------------
            # EE -> JOINTS
            # ----------------------------------------------------

            joint_action = ee_to_joint_processor(
                (ee_action, robot_obs)
            )

            # ----------------------------------------------------
            # Print IK result
            # ----------------------------------------------------

            print()
            print("IK calculated joint action:")

            for name, value in joint_action.items():
                print(f"  {name:15s}: {value}")

            # ----------------------------------------------------
            # Send calculated joints to robot
            # ----------------------------------------------------

            if not args.dry_run:

                robot.send_action(joint_action)

            # ----------------------------------------------------
            # Maintain dataset timing
            # ----------------------------------------------------

            elapsed = time.perf_counter() - frame_start

            frame_period = 1.0 / dataset.fps

            sleep_time = frame_period - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:

        if not args.dry_run:

            print()
            print("Disconnecting robot...")

            robot.disconnect()

        print()
        print("=" * 70)
        print("Replay finished.")
        print("=" * 70)


if __name__ == "__main__":
    main()