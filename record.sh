#!/bin/bash

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=mind_blowing_mandy \
  --robot.cameras="{arm: {type: opencv, index_or_path: '/dev/video10', width: 640, height: 480, fps: 30, fourcc: MJPG}, overhead: {type: opencv, index_or_path: '/dev/video8', width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=tinkering_tilda \
  --display_data=true \
  --dataset.fps=30 \
  --dataset.num_episodes=50 \
  --dataset.single_task="Push and pick up the metal instruments to separate them" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.episode_time_s=600 \
  --dataset.repo_id=aboardman/record-50-separate_instruments_2.1
