import argparse
import shutil
from pathlib import Path


IMPORT_LINE = "import openpi.policies.rlbench_waypoint_policy as rlbench_waypoint_policy\n"

DATA_CONFIG_CLASS = r'''

@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchWaypointDataConfig(DataConfigFactory):
    """Data config for RLBench full-task sparse-waypoint pi0.5 baseline."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/front_image": "front_image",
                        "observation/left_shoulder_image": "left_shoulder_image",
                        "observation/right_shoulder_image": "right_shoulder_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[rlbench_waypoint_policy.RlbenchWaypointInputs(model_type=model_config.model_type)],
            outputs=[rlbench_waypoint_policy.RlbenchWaypointOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
'''

TRAIN_CONFIG_ENTRY = r'''
    TrainConfig(
        name="pi05_rlbench_waypoint_h1",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=1, discrete_state_input=False),
        data=LeRobotRLBenchWaypointDataConfig(
            repo_id="rlbench/selected10_pi05_waypoint_h1",
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Main first-stage setting for the 8x A100 40GB π0.5-Full-3V recipe.
        # See /raid/home/than/zhiyuan/rlbench_baseline_hyperparams.md.
        batch_size=128,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        save_interval=2_000,
        keep_period=2_000,
        fsdp_devices=8,
    ),
'''


def replace_existing_train_config(text: str) -> tuple[str, bool]:
    name_idx = text.find('name="pi05_rlbench_waypoint_h1"')
    if name_idx < 0:
        return text, False

    start = text.rfind("    TrainConfig(", 0, name_idx)
    if start < 0:
        raise RuntimeError("Found RLBench config name but could not find TrainConfig start")

    depth = 0
    end = None
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                if end < len(text) and text[end] == ",":
                    end += 1
                if end < len(text) and text[end] == "\n":
                    end += 1
                break

    if end is None:
        raise RuntimeError("Could not find end of RLBench TrainConfig block")

    replacement = TRAIN_CONFIG_ENTRY.strip("\n") + "\n"
    if text[start:end] == replacement:
        return text, False
    return text[:start] + replacement + text[end:], True


def build_parser():
    parser = argparse.ArgumentParser(description="Patch OpenPI with the RLBench waypoint pi0.5 config.")
    parser.add_argument("--openpi-dir", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--no-backup", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    openpi_dir = Path(args.openpi_dir).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path(__file__).resolve().parents[1]

    policy_src = repo_root / "openpi_extensions" / "rlbench_waypoint_policy.py"
    policy_dst = openpi_dir / "src" / "openpi" / "policies" / "rlbench_waypoint_policy.py"
    config_path = openpi_dir / "src" / "openpi" / "training" / "config.py"
    if not policy_src.exists():
        raise FileNotFoundError(policy_src)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    shutil.copy2(policy_src, policy_dst)

    text = config_path.read_text(encoding="utf-8")
    changed = False
    if IMPORT_LINE not in text:
        marker = "import openpi.policies.libero_policy as libero_policy\n"
        if marker not in text:
            raise RuntimeError("Could not find import marker in OpenPI config.py")
        text = text.replace(marker, marker + IMPORT_LINE)
        changed = True

    if "class LeRobotRLBenchWaypointDataConfig" not in text:
        marker = "\n@dataclasses.dataclass(frozen=True)\nclass RLDSDroidDataConfig"
        if marker not in text:
            raise RuntimeError("Could not find data config insertion marker in OpenPI config.py")
        text = text.replace(marker, DATA_CONFIG_CLASS + marker)
        changed = True

    text, replaced_existing = replace_existing_train_config(text)
    changed = changed or replaced_existing

    if 'name="pi05_rlbench_waypoint_h1"' not in text:
        marker = None
        for candidate in (
            "    #\n    # Fine-tuning Aloha configs.",
            "    #\n    # Aloha training configs.",
        ):
            if candidate in text:
                marker = candidate
                break
        if marker is None:
            raise RuntimeError("Could not find train config insertion marker in OpenPI config.py")
        text = text.replace(marker, TRAIN_CONFIG_ENTRY + "\n" + marker)
        changed = True

    if changed:
        if not args.no_backup:
            backup = config_path.with_suffix(".py.bak_before_rlbench_waypoint")
            if not backup.exists():
                shutil.copy2(config_path, backup)
        config_path.write_text(text, encoding="utf-8")

    print(f"Copied policy transform: {policy_dst}")
    print(f"Patched config: {config_path}")
    print("Config name: pi05_rlbench_waypoint_h1")


if __name__ == "__main__":
    main()
