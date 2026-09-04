#!/usr/bin/env python3

"""
Convert an existing LeRobot SO-101 dataset from joint-space to end-effector (EE) space.

Input:
    action:
        shoulder_pan.pos
        shoulder_lift.pos
        elbow_flex.pos
        wrist_flex.pos
        wrist_roll.pos
        gripper.pos

    observation.state:
        same six joint values

Output:
    action:
        ee.x
        ee.y
        ee.z
        ee.wx
        ee.wy
        ee.wz
        ee.gripper_pos

    observation.state:
        ee.x
        ee.y
        ee.z
        ee.wx
        ee.wy
        ee.wz
        ee.gripper_pos

All images, tasks, timestamps, episode boundaries, etc. are preserved.
The original dataset is NOT modified.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIPPER_NAME = "gripper"
EE_NAMES = ["x", "y", "z", "wx", "wy", "wz", "gripper_pos"]


def joints_to_ee(joint_values, kinematics):
    """Convert SO-101 joint positions to EE pose."""
    joint_values = np.asarray(joint_values, dtype=np.float64)
    transform = kinematics.forward_kinematics(joint_values)
    position = transform[:3, 3]

    from scipy.spatial.transform import Rotation
    rotation_vector = Rotation.from_matrix(transform[:3, :3]).as_rotvec()

    return np.array([position[0], position[1], position[2], rotation_vector[0], rotation_vector[1], rotation_vector[2]], dtype=np.float32)


def convert_frame_vector(values, kinematics):
    """Convert [5 arm joints, gripper] into [x, y, z, wx, wy, wz, gripper]."""
    values = np.asarray(values, dtype=np.float64)
    arm_joints = values[:5]
    gripper = values[5]
    ee_pose = joints_to_ee(arm_joints, kinematics)
    return np.concatenate([ee_pose, np.array([gripper], dtype=np.float32)]).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_repo", type=str, required=True, help="Existing Hugging Face dataset repo ID.")
    parser.add_argument("--output_repo", type=str, required=True, help="New Hugging Face dataset repo ID.")
    parser.add_argument("--urdf", type=str, required=True, help="Path to SO101 new calibration URDF.")
    parser.add_argument("--revision", type=str, default=None, help="Optional dataset revision.")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="Optional episode numbers to convert. If omitted, converts the entire dataset.")
    parser.add_argument("--push_to_hub", action="store_true", help="Push converted dataset to Hugging Face.")
    args = parser.parse_args()

    urdf_path = Path(args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"Could not find URDF:\n{urdf_path}")

    print("\n" + "=" * 70)
    print("LOADING DATASET")
    print("=" * 70)
    print(f"Input dataset : {args.input_repo}")

    source_dataset = LeRobotDataset(repo_id=args.input_repo, revision=args.revision, download_videos=True)

    print(f"Episodes      : {source_dataset.meta.total_episodes}")
    print(f"Frames        : {source_dataset.meta.total_frames}")
    print(f"FPS           : {source_dataset.fps}")

    required_action = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]
    required_state = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]

    action_names = source_dataset.meta.features["action"]["names"]
    state_names = source_dataset.meta.features["observation.state"]["names"]

    print("\nAction features:")
    print(action_names)
    print("\nObservation features:")
    print(state_names)

    for name in required_action:
        if name not in action_names:
            raise ValueError(f"Missing required action feature: {name}")

    for name in required_state:
        if name not in state_names:
            raise ValueError(f"Missing required observation feature: {name}")

    print("\n" + "=" * 70)
    print("INITIALIZING SO-101 KINEMATICS")
    print("=" * 70)
    print(f"URDF: {urdf_path}")

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=JOINT_NAMES,
    )

    print("Target frame: gripper_frame_link")
    print("Joint order:")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  {i}: {name}")

    output_features = {}

    for key, feature in source_dataset.meta.features.items():
        if key in ["action", "observation.state"]:
            continue
        output_features[key] = feature.copy()

    output_features["action"] = {
        "dtype": "float32",
        "shape": [7],
        "names": [f"ee.{name}" for name in EE_NAMES],
    }

    output_features["observation.state"] = {
        "dtype": "float32",
        "shape": [7],
        "names": [f"ee.{name}" for name in EE_NAMES],
    }

    print("\n" + "=" * 70)
    print("OUTPUT FEATURES")
    print("=" * 70)
    print("action:")
    print(output_features["action"])
    print("\nobservation.state:")
    print(output_features["observation.state"])

    print("\n" + "=" * 70)
    print("CREATING OUTPUT DATASET")
    print("=" * 70)

    output_dataset = LeRobotDataset.create(
        repo_id=args.output_repo,
        fps=source_dataset.fps,
        features=output_features,
        robot_type=source_dataset.meta.robot_type,
        use_videos=True,
    )

    if args.episodes is None:
        episodes = range(source_dataset.meta.total_episodes)
    else:
        episodes = args.episodes

    episodes = list(episodes)
    print(f"Converting {len(episodes)} episodes.")

    total_frames = 0

    for episode_number in episodes:
        print(f"\nEpisode {episode_number} ({len(episodes)} total)")

        episode = source_dataset.meta.episodes[episode_number]
        from_idx = int(episode["dataset_from_index"])
        to_idx = int(episode["dataset_to_index"])
        episode_length = to_idx - from_idx

        print(f"  Frames: {episode_length}")

        for dataset_index in range(from_idx, to_idx):
            frame = source_dataset[dataset_index]

            action = frame["action"]
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action_ee = convert_frame_vector(action, kinematics)

            observation = frame["observation.state"]
            if isinstance(observation, torch.Tensor):
                observation = observation.detach().cpu().numpy()
            observation_ee = convert_frame_vector(observation, kinematics)

            output_frame = {}

            for key in source_dataset.meta.features:
                if key in ["action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"]:
                    continue

                if key in frame:
                    value = frame[key]
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                        # If it's an image (3 channels), and CHW, convert to HWC
                        if key.startswith("observation.image") and value.ndim == 3 and value.shape[0] == 3:
                            value = np.transpose(value, (1, 2, 0))
                    output_frame[key] = value

            output_frame["action"] = action_ee
            output_frame["observation.state"] = observation_ee

            if "task" in frame:
                task = frame["task"]
                if isinstance(task, torch.Tensor):
                    task = task.item()
                output_frame["task"] = task
            else:
                raise RuntimeError("Frame does not contain task information.")

            output_dataset.add_frame(output_frame)

            total_frames += 1

            if total_frames % 1000 == 0:
                print(f"    Converted {total_frames:,} frames")

        output_dataset.save_episode()
        print(f"  ✓ Episode {episode_number} saved")

    print("\n" + "=" * 70)
    print("FINALIZING DATASET")
    print("=" * 70)

    output_dataset.finalize()

    print(f"\nConverted frames: {total_frames:,}")
    print(f"Output dataset : {args.output_repo}")

    if args.push_to_hub:
        print("\n" + "=" * 70)
        print("PUSHING TO HUGGING FACE")
        print("=" * 70)

        output_dataset.push_to_hub()
        print("\n✓ Dataset pushed successfully.")
    else:
        print("\nDataset was NOT pushed to Hugging Face.")
        print("Run again with --push_to_hub when ready.")


if __name__ == "__main__":
    main()
