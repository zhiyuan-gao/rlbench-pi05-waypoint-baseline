import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .common import (
    VIEW_NAMES,
    absolute_rpy7_from_obs,
    clean_waypoints,
    filter_manifest_rows,
    image_path_for_frame,
    load_observations,
    obs_to_state,
    parse_root_mapping,
    read_jsonl,
    resolve_episode_dir,
    write_json,
)


def _load_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if image_size > 0 and img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _current_frames_for_segment(start: int, target: int, sample_every_n: int) -> list[int]:
    if sample_every_n <= 0:
        return [int(start)]
    frames = list(range(int(start), int(target), int(sample_every_n)))
    if int(start) not in frames:
        frames.insert(0, int(start))
    return sorted(set(f for f in frames if int(start) <= f < int(target)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert RLBench selected10 full-task heuristic waypoints to a LeRobot dataset for OpenPI/pi0.5."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-id", default="rlbench/selected10_pi05_waypoint_h1")
    parser.add_argument("--robot-type", default="panda")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--split", default="train", choices=("train", "val", "test", "all"))
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--rgb-root-200", default=None)
    parser.add_argument("--rgb-root-400", default=None)
    parser.add_argument("--rgb-root", action="append", default=[])
    parser.add_argument("--lowdim-root-200", default=None)
    parser.add_argument("--lowdim-root-400", default=None)
    parser.add_argument("--lowdim-root", action="append", default=[])
    parser.add_argument("--state-mode", choices=("ee_rotvec", "ee_rpy"), default="ee_rotvec")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--sample-every-n", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-episodes-per-task", type=int, default=None)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--image-writer-processes", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-image-paths", action="store_true")
    parser.add_argument("--summary-out", default=None)
    return parser


def main(argv=None):
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:
        raise RuntimeError(
            "Could not import LeRobotDataset. Run this inside the OpenPI uv environment "
            "after installing OpenPI."
        ) from exc

    args = build_parser().parse_args(argv)
    rgb_roots = parse_root_mapping(args.rgb_root_200, args.rgb_root_400, args.rgb_root, root_name="RGB root")
    lowdim_roots = parse_root_mapping(
        args.lowdim_root_200,
        args.lowdim_root_400,
        args.lowdim_root,
        root_name="low-dim root",
    )
    split = None if args.split == "all" else args.split
    rows = filter_manifest_rows(
        read_jsonl(Path(args.manifest)),
        split=split,
        tasks=args.task,
        max_episodes=args.max_episodes,
        max_episodes_per_task=args.max_episodes_per_task,
    )

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=int(args.fps),
        features={
            "front_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "left_shoulder_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "right_shoulder_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=int(args.image_writer_threads),
        image_writer_processes=int(args.image_writer_processes),
    )

    counts = Counter()
    examples = []
    total_frames = 0
    for row in tqdm(rows, desc="Converting episodes"):
        rgb_episode_dir = resolve_episode_dir(row, rgb_roots, root_name="RGB root")
        lowdim_episode_dir = resolve_episode_dir(row, lowdim_roots, root_name="low-dim root")
        observations = load_observations(lowdim_episode_dir, loose=True)
        num_frames = min(int(row.get("num_frames", len(observations))), len(observations))
        waypoints = clean_waypoints(row, num_frames=num_frames)
        points = [0] + [p for p in waypoints if 0 < int(p) < num_frames]
        points = sorted(set(int(p) for p in points))
        if len(points) < 2:
            continue

        task_text = str(row.get("task_instruction") or row.get("task") or "").strip()
        if not task_text:
            task_text = str(row.get("task", "")).replace("_", " ")

        episode_frames = 0
        for segment_idx in range(len(points) - 1):
            start = int(points[segment_idx])
            target = int(points[segment_idx + 1])
            if target <= start:
                continue
            for current in _current_frames_for_segment(start, target, int(args.sample_every_n)):
                if args.validate_image_paths:
                    for view in VIEW_NAMES:
                        path = image_path_for_frame(rgb_episode_dir, view, current)
                        if not path.exists():
                            raise FileNotFoundError(path)
                obs = observations[current]
                target_obs = observations[target]
                dataset.add_frame(
                    {
                        "front_image": _load_image(image_path_for_frame(rgb_episode_dir, "front", current), args.image_size),
                        "left_shoulder_image": _load_image(
                            image_path_for_frame(rgb_episode_dir, "left_shoulder", current), args.image_size
                        ),
                        "right_shoulder_image": _load_image(
                            image_path_for_frame(rgb_episode_dir, "right_shoulder", current), args.image_size
                        ),
                        "state": obs_to_state(obs, mode=args.state_mode).astype(np.float32),
                        "actions": absolute_rpy7_from_obs(target_obs).astype(np.float32),
                        "task": task_text,
                    }
                )
                total_frames += 1
                episode_frames += 1
                counts[(row.get("split"), row.get("task"))] += 1
                if len(examples) < 8:
                    examples.append(
                        {
                            "task": row.get("task"),
                            "split": row.get("split"),
                            "source_bundle": row.get("source_bundle"),
                            "episode": row.get("episode"),
                            "current_frame_idx": int(current),
                            "target_frame_idx": int(target),
                            "task_instruction": task_text,
                        }
                    )
        if episode_frames:
            dataset.save_episode()

    summary = {
        "repo_id": args.repo_id,
        "output_path": str(output_path),
        "manifest": str(Path(args.manifest).resolve()),
        "split_filter": args.split,
        "tasks_filter": args.task,
        "state_mode": args.state_mode,
        "action_format": "absolute_rpy7",
        "action_horizon": 1,
        "sample_every_n": int(args.sample_every_n),
        "num_manifest_episodes": len(rows),
        "num_lerobot_frames": int(total_frames),
        "counts": {f"{split}:{task}": int(value) for (split, task), value in sorted(counts.items())},
        "examples": examples,
    }
    summary_out = Path(args.summary_out) if args.summary_out else output_path / "conversion_summary.json"
    write_json(summary_out, summary)
    print(f"Wrote LeRobot dataset: {output_path}")
    print(f"Wrote summary: {summary_out}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
