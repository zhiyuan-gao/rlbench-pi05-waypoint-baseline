import argparse
import json

import numpy as np


def _shape_dtype(value):
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke test a converted RLBench pi0.5 LeRobot dataset.")
    parser.add_argument("--repo-id", default="rlbench/selected10_pi05_waypoint_h1")
    parser.add_argument("--num-samples", type=int, default=4)
    return parser


def main(argv=None):
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    except Exception as exc:
        raise RuntimeError("Could not import LeRobot. Run this inside the OpenPI uv environment.") from exc

    args = build_parser().parse_args(argv)
    metadata = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(args.repo_id)
    print(f"HF_LEROBOT_HOME={HF_LEROBOT_HOME}")
    print(f"repo_id={args.repo_id}")
    print(f"num_frames={len(dataset)}")
    print(f"fps={metadata.fps}")
    print(f"features={json.dumps(metadata.features, indent=2, sort_keys=True, default=str)}")

    for idx in range(min(int(args.num_samples), len(dataset))):
        item = dataset[idx]
        summary = {"idx": idx}
        for key in ("front_image", "left_shoulder_image", "right_shoulder_image", "state", "actions"):
            if key in item:
                summary[key] = _shape_dtype(item[key])
        if "task" in item:
            summary["task"] = str(item["task"])
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

