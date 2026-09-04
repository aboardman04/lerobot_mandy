---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).


<a class="flex" href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=aboardman/my_giant_combined_dataset2-5-6-8-9-10-rep">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl.svg"/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/badges/resolve/main/visualize-this-dataset-xl-dark.svg"/>
</a>


## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v3.0",
    "fps": 30,
    "features": {
        "action": {
            "dtype": "float32",
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos"
            ],
            "shape": [
                6
            ]
        },
        "observation.state": {
            "dtype": "float32",
            "names": [
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos"
            ],
            "shape": [
                6
            ]
        },
        "observation.images.arm": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false,
                "video.g": null,
                "video.crf": null,
                "video.preset": null,
                "video.fast_decode": null,
                "video.video_backend": "pyav",
                "video.extra_options": {}
            }
        },
        "observation.images.overhead": {
            "dtype": "video",
            "shape": [
                480,
                640,
                3
            ],
            "names": [
                "height",
                "width",
                "channels"
            ],
            "info": {
                "video.height": 480,
                "video.width": 640,
                "video.codec": "av1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": false,
                "video.fps": 30,
                "video.channels": 3,
                "has_audio": false,
                "video.g": null,
                "video.crf": null,
                "video.preset": null,
                "video.fast_decode": null,
                "video.video_backend": "pyav",
                "video.extra_options": {}
            }
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null
        }
    },
    "total_episodes": 363,
    "total_frames": 272652,
    "total_tasks": 2,
    "chunks_size": 1000,
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 200,
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "robot_type": "so_follower",
    "splits": {
        "train": "0:363"
    }
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```