import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


VIEW_NAMES = ("front", "left_shoulder", "right_shoulder")


class DummyRlbenchObject:
    """Fallback object for reading RLBench pickle files without importing RLBench."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class LooseUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("rlbench") or module.startswith("pyrep"):
            return DummyRlbenchObject
        return super().find_class(module, name)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_tasks(tasks: Optional[Sequence[str]]) -> Optional[set]:
    if not tasks:
        return None
    parsed = set()
    for item in tasks:
        for task in str(item).replace(",", " ").split():
            if task.strip():
                parsed.add(task.strip())
    return parsed or None


def parse_root_mapping(root_200=None, root_400=None, extra_roots=None, root_name: str = "root"):
    roots = {}
    if root_200:
        roots["all200"] = Path(root_200).resolve()
    if root_400:
        roots["all400"] = Path(root_400).resolve()
    for item in extra_roots or []:
        if "=" not in item:
            raise ValueError(f"{root_name} entries must have form source_bundle=/path")
        key, value = item.split("=", 1)
        roots[key.strip()] = Path(value).resolve()
    if not roots:
        raise ValueError(f"At least one {root_name} is required.")
    return roots


def filter_manifest_rows(
    rows: Sequence[dict],
    split: Optional[str],
    tasks: Optional[Sequence[str]],
    max_episodes: Optional[int] = None,
    max_episodes_per_task: Optional[int] = None,
) -> List[dict]:
    task_filter = parse_tasks(tasks)
    selected = []
    per_task = defaultdict(int)
    for row in rows:
        task = row.get("task")
        if split is not None and row.get("split") != split:
            continue
        if task_filter is not None and task not in task_filter:
            continue
        if max_episodes is not None and len(selected) >= int(max_episodes):
            break
        if max_episodes_per_task is not None and per_task[task] >= int(max_episodes_per_task):
            continue
        selected.append(row)
        per_task[task] += 1
    if not selected:
        raise RuntimeError("No manifest rows matched the requested filters.")
    return selected


def source_bundle_to_root(row: dict, roots: Dict[str, Path], root_name: str) -> Path:
    bundle = str(row.get("source_bundle", "all200"))
    if bundle not in roots:
        known = ", ".join(sorted(roots))
        raise KeyError(f"source_bundle={bundle!r} has no {root_name}. Known: {known}")
    return roots[bundle]


def resolve_episode_dir(row: dict, roots: Dict[str, Path], root_name: str) -> Path:
    root = source_bundle_to_root(row, roots, root_name)
    rel = row.get("rgb_episode_relpath")
    if rel:
        return Path(root) / rel
    return Path(root) / row["task"] / row["variation"] / "episodes" / row["episode"]


def load_observations(episode_dir: Path, loose: bool = True):
    with (Path(episode_dir) / "low_dim_obs.pkl").open("rb") as f:
        demo = LooseUnpickler(f).load() if loose else pickle.load(f)
    observations = getattr(demo, "_observations", demo)
    if not isinstance(observations, (list, tuple)):
        raise ValueError(f"{episode_dir}/low_dim_obs.pkl did not contain observations")
    return observations


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.clip(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12, None)


def quat_to_rotvec(q):
    q = normalize_quat(q)
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz = q[..., :3]
    w = np.clip(q[..., 3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.full_like(angle, 2.0)
    np.divide(angle, sin_half, out=scale, where=sin_half > 1e-8)
    return xyz * scale[..., None]


def rotvec_to_quat(rotvec):
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    half = 0.5 * angle
    scale = np.full_like(angle, 0.5)
    np.divide(np.sin(half), angle, out=scale, where=angle > 1e-8)
    xyz = rotvec * scale
    w = np.cos(half)
    return normalize_quat(np.concatenate([xyz, w], axis=-1))


def gripper_open_value(obs, threshold: float = 0.95) -> float:
    joint_positions = getattr(obs, "gripper_joint_positions", None)
    if joint_positions is None:
        return float(getattr(obs, "gripper_open", 0.0))
    joint_positions = np.asarray(joint_positions, dtype=np.float32)
    if joint_positions.size == 0:
        return float(getattr(obs, "gripper_open", 0.0))
    return 1.0 if float(joint_positions[0]) / 0.04 > float(threshold) else 0.0


def obs_to_state(obs, mode: str = "ee_rotvec") -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    if ee_pose.shape != (7,):
        raise ValueError(f"Expected obs.gripper_pose shape (7,), got {ee_pose.shape}")
    gripper = np.asarray([gripper_open_value(obs)], dtype=np.float32)
    if mode == "ee_rotvec":
        return np.concatenate([ee_pose[:3], quat_to_rotvec(ee_pose[3:7]).astype(np.float32), gripper])
    raise ValueError(f"Unsupported state mode: {mode}")


def absolute_rotvec7_from_obs(obs) -> np.ndarray:
    ee_pose = np.asarray(obs.gripper_pose, dtype=np.float32)
    rotvec = quat_to_rotvec(ee_pose[3:7]).astype(np.float32)
    return np.concatenate([ee_pose[:3], rotvec, [gripper_open_value(obs)]]).astype(np.float32)


def clean_waypoints(row: dict, num_frames: Optional[int] = None) -> List[int]:
    raw = row.get("full_task_heuristic_waypoints")
    if raw is None:
        raw = row.get("clean_keypoints") or row.get("keypoints") or row.get("event_all_grouping_keypoints")
    if raw is None:
        raise KeyError("Manifest row has no full_task_heuristic_waypoints / clean_keypoints / keypoints field")
    max_frame = None if num_frames is None else max(int(num_frames) - 1, 0)
    points = []
    for point in raw:
        point = int(point)
        if point <= 0:
            continue
        if max_frame is not None:
            point = min(max(point, 0), max_frame)
        points.append(point)
    return sorted(set(points))


def image_path_for_frame(episode_dir: Path, view: str, frame_idx: int) -> Path:
    return Path(episode_dir) / f"{view}_rgb" / f"{int(frame_idx)}.png"
