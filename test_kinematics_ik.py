import numpy as np
from lerobot.model.kinematics import RobotKinematics

kinematics = RobotKinematics(
    urdf_path="/home/aboardman/lerobot/SO101/so101_new_calib.urdf",
    target_frame_name="gripper_frame_link",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
)

# Pass 5 joints
q = np.array([0., 0., 0., 0., 0.])
des = np.eye(4)
kinematics.inverse_kinematics(q, des)
