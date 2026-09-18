#!/usr/bin/env python

"""
Rollout an EE-space policy on an SO101/SO100 robot.

The policy operates in end-effector space:

    robot joints
        ↓
    Forward Kinematics
        ↓
    EE observation
        ↓
       policy
        ↓
    EE action
        ↓
    Inverse Kinematics
        ↓
    robot joints
        ↓
       robot

Example:

python rollout_ee.py \
    --policy.path=aboardman/smolVLA_ee_cube_1 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=mind_blowing_mandy \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="lift the white cube" \
    --duration=60 \
    --fps=30 \
    --urdf_path=./SO101/so101_new_calib.urdf
"""

import logging

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401

from lerobot.configs import parser
from lerobot.configs import PreTrainedConfig

from lerobot.types import RobotAction, RobotObservation

from lerobot.model.kinematics import RobotKinematics

from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)

from lerobot.robots.so_follower.robot_kinematic_processor import (
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)

from dataclasses import dataclass
from lerobot.rollout import (
    BaseStrategyConfig,
    RolloutConfig,
    build_rollout_context,
)

from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.strategies import BaseStrategy

from lerobot.utils.process import ProcessSignalHandler
from lerobot.utils.utils import init_logging


logger = logging.getLogger(__name__)


@dataclass
class EERolloutConfig(RolloutConfig):
    urdf_path: str = ""
    ee_frame: str = "gripper_frame_link"

@parser.wrap()
def rollout(cfg: EERolloutConfig):
    """
    Run an EE-space policy on the robot.

    The policy receives EE observations and produces EE actions.
    The processors convert:

        joint → EE

    before policy inference, and:

        EE → joint

    before sending the action to the robot.
    """

    init_logging()

    # ------------------------------------------------------------
    # 1. Basic configuration
    # ------------------------------------------------------------

    logger.info("Starting EE-space rollout")

    logger.info("Policy: %s", cfg.policy.pretrained_path)
    logger.info("Robot: %s", cfg.robot.type)
    logger.info("Task: %s", cfg.task)
    logger.info("Duration: %s seconds", cfg.duration)
    logger.info("FPS: %s", cfg.fps)

    # ------------------------------------------------------------
    # 2. Robot
    # ------------------------------------------------------------

    if not isinstance(cfg.robot, SO100FollowerConfig):
        raise ValueError(
            "This EE rollout currently expects an SO100Follower/SO101 "
            "robot configuration."
        )

    robot_config = cfg.robot

    # ------------------------------------------------------------
    # 3. Get robot motor names
    # ------------------------------------------------------------
    #
    # The RobotKinematics class needs the names of the robot joints.
    #
    # We create a temporary robot object only to inspect the motor
    # names. The actual robot connection is managed later by
    # build_rollout_context().
    # ------------------------------------------------------------

    logger.info("Inspecting robot motor names...")

    temp_robot = SO100Follower(robot_config)

    motor_names = list(temp_robot.bus.motors.keys())

    logger.info("Robot motor names: %s", motor_names)

    # ------------------------------------------------------------
    # 4. Build kinematics solver
    # ------------------------------------------------------------

    logger.info("Loading URDF: %s", cfg.urdf_path)

    kinematics_solver = RobotKinematics(
        urdf_path=cfg.urdf_path,
        target_frame_name=cfg.ee_frame,
        joint_names=motor_names,
    )

    logger.info(
        "Kinematics configured with EE frame: %s",
        cfg.ee_frame,
    )

    # ------------------------------------------------------------
    # 5. Joint → EE observation processor
    # ------------------------------------------------------------
    #
    # Robot hardware produces:
    #
    #     joint positions
    #
    # We convert them into:
    #
    #     EE position/orientation
    #
    # before giving the observation to the policy.
    # ------------------------------------------------------------

    robot_joints_to_ee_pose_processor = RobotProcessorPipeline[
        RobotObservation,
        RobotObservation
    ](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # ------------------------------------------------------------
    # 6. EE → joint action processor
    # ------------------------------------------------------------
    #
    # Policy outputs:
    #
    #     EE action
    #
    # IK converts it into:
    #
    #     joint action
    #
    # which is then sent to the SO101.
    # ------------------------------------------------------------

    robot_ee_to_joints_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation],
        RobotAction
    ](
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ------------------------------------------------------------
    # 6b. Teleop -> EE action processor (for dataset features)
    # ------------------------------------------------------------
    #
    # We need to tell the framework that the dataset features (which
    # correspond to the policy output) are in EE space, not joint space.
    # ------------------------------------------------------------

    teleop_joints_to_ee_processor = RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation],
        RobotAction
    ](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics_solver,
                motor_names=motor_names,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # ------------------------------------------------------------
    # 7. Load policy configuration
    # ------------------------------------------------------------

    logger.info(
        "Using policy configuration: %s",
        cfg.policy.pretrained_path,
    )

    policy_config = cfg.policy

    from lerobot.configs.types import FeatureType
    expected_visuals = [
        k for k, v in policy_config.input_features.items() if v.type == FeatureType.VISUAL
    ]
    provided_visuals = list(robot_config.cameras.keys())
    
    if not cfg.rename_map:
        cfg.rename_map = {}
        for prov, exp in zip(provided_visuals, expected_visuals):
            cfg.rename_map[f"observation.images.{prov}"] = exp
            logger.info("Auto-mapping camera %s to %s", prov, exp)
        
        missing = len(expected_visuals) - len(provided_visuals)
        if missing > 0:
            if hasattr(policy_config, "empty_cameras"):
                policy_config.empty_cameras = missing
                logger.info("Setting empty_cameras=%d to handle missing visuals", missing)
            else:
                logger.warning("Policy lacks empty_cameras support, but %d cameras are missing.", missing)

    # ------------------------------------------------------------
    # 8. Rollout configuration
    # ------------------------------------------------------------

    rollout_cfg = RolloutConfig(
        robot=robot_config,
        policy=policy_config,
        strategy=BaseStrategyConfig(),
        inference=SyncInferenceConfig(),
        fps=cfg.fps,
        duration=cfg.duration,
        task=cfg.task,
        rename_map=cfg.rename_map,
    )

    # ------------------------------------------------------------
    # 9. Shutdown handling
    # ------------------------------------------------------------

    signal_handler = ProcessSignalHandler(
        use_threads=True,
        display_pid=False,
    )

    shutdown_event = signal_handler.shutdown_event

    # ------------------------------------------------------------
    # 10. Build rollout context
    # ------------------------------------------------------------
    #
    # This is the key part.
    #
    # We tell LeRobot:
    #
    #   robot observation → FK → EE → policy
    #
    # and:
    #
    #   policy EE action → IK → robot joint action
    #
    # ------------------------------------------------------------

    logger.info("Building rollout context...")

    # Mock input to avoid interactive calibration prompt in non-interactive environment
    import builtins
    original_input = builtins.input
    def mock_input(prompt=""):
        logger.info(f"Mocking input for prompt: {prompt}")
        return ""
    builtins.input = mock_input

    ctx = build_rollout_context(
        rollout_cfg,
        shutdown_event,

        teleop_action_processor=teleop_joints_to_ee_processor,
        robot_action_processor=robot_ee_to_joints_processor,

        robot_observation_processor=robot_joints_to_ee_pose_processor,
    )
    builtins.input = original_input

    # Patch ordered action keys for EE space since context.py hardcodes to HW joints
    ee_keys = ctx.data.dataset_features["action"]["names"]
    ctx.data.ordered_action_keys = ee_keys
    if hasattr(ctx.policy.inference, "_ordered_action_keys"):
        ctx.policy.inference._ordered_action_keys = ee_keys

    # ------------------------------------------------------------
    # 11. Create rollout strategy
    # ------------------------------------------------------------

    strategy = BaseStrategy(rollout_cfg.strategy)

    logger.info("Starting rollout")

    try:
        strategy.setup(ctx)
        strategy.run(ctx)

    except KeyboardInterrupt:
        logger.info("Rollout interrupted by user")

    finally:
        logger.info("Stopping robot")
        strategy.teardown(ctx)

    logger.info("Rollout finished")


def main():
    """
    CLI entry point.
    """

    rollout()


if __name__ == "__main__":
    main()