# RLBench pi0.5 Full-Task Waypoint Baseline

这是一个给 RLBench selected10 heuristic-waypoint imitation 使用的 OpenPI / pi0.5 baseline sidecar repo。

这个 repo 和 Wan/VPP 代码、Diffusion Policy baseline 代码都是分开的。它不使用 Wan hidden tokens，不使用视频模型特征，不使用 event/subtask boundary，也不使用 dense planner trajectory supervision。模型使用：

- `front`、`left_shoulder`、`right_shoulder` 三视角 raw RGB
- full-task language instruction
- end-effector pose + gripper state proprio
- pi0.5 / OpenPI flow-matching action expert
- 稀疏动作目标：预测下一个 full-task heuristic waypoint

动作格式是 `absolute_rpy7`：`x, y, z, roll, pitch, yaw, gripper_open`。训练时 `action_horizon=1`，online eval 时每次预测一个 waypoint，并交给 RLBench planner/IK action mode 执行。

## 数据格式

repo 内置了 selected10 train100/val25/test25 manifest：

```text
manifests/selected10_fulltask_heuristic_waypoints_train100_val25_test25_from_train450_stratified_20260606.jsonl
```

每一行是一个 episode，转换代码需要这些字段：

- `split`: `train`、`val` 或 `test`
- `source_bundle`: `all200` 或 `all400`
- `rgb_episode_relpath`: 例如 `{task}/variation0/episodes/episode0`
- `task`、`variation`、`episode`
- `task_instruction`
- `full_task_heuristic_waypoints`
- `num_frames`

RGB 路径会这样重建：

```text
{rgb_root_for_source_bundle}/{rgb_episode_relpath}/{view}_rgb/{frame}.png
```

low-dim observation 会从这里读取：

```text
{lowdim_root_for_source_bundle}/{rgb_episode_relpath}/low_dim_obs.pkl
```

转换后的 LeRobot frame 是 sparse waypoint-call 样本。默认只在 full-task waypoint graph 的节点上采样：

```text
t=0          -> target = waypoint_0
t=waypoint_i -> target = waypoint_{i+1}
```

例如某条 demo 的 full-task heuristic waypoints 是 `[36, 49, 61, 92, 108]`：

```text
t=0   -> target frame 36
t=36  -> target frame 49
t=49  -> target frame 61
t=61  -> target frame 92
t=92  -> target frame 108
```

每条 demo 仍然是一个 variable-length LeRobot episode，但训练样本是 episode 里的 frame。不同 demo 的 waypoint 数不同没有关系，因为每个 frame 的 tensor shape 固定：

```text
front_image            uint8   [256, 256, 3]
left_shoulder_image    uint8   [256, 256, 3]
right_shoulder_image   uint8   [256, 256, 3]
state                  float32 [7]
actions                float32 [7]
task                   string
```

## HPC 环境要求

这个 repo 是 OpenPI 的 sidecar，不 vendor OpenPI 本体。暂定训练机器是 8x A100 40GB。HPC 上需要：

- `git`
- `git-lfs`
- `uv`
- Python 3.10/3.11 compatible environment for this sidecar package
- OpenPI 的 CUDA/JAX/PyTorch dependencies，由安装脚本创建在 OpenPI `.venv`
- 两个 RGB root：`all200` 和 `all400`
- 两个 low-dim metadata root：`all200` 和 `all400`

训练参数参考 `/raid/home/than/zhiyuan/rlbench_baseline_hyperparams.md` 里的 π0.5-Full-3V 设置，但考虑到当前 selected10 sparse-waypoint 数据量偏小，主训练先采用更温和的 staged schedule。这里的 `batch_size` 是 OpenPI/JAX 的 global batch size，不是每卡 batch size。OpenPI `scripts/train.py` 要求 global batch 能被 JAX device count 整除；8 卡单节点时，global 128 大约等于每卡 16 个样本 shard：

```text
global batch_size = 128
8 GPUs            = about 16 samples per GPU
fsdp_devices      = 8
num_train_steps   = 20000     # first-stage run
max_train_steps   = 60000     # only if validation keeps improving
warmup_steps      = 10000
learning_rate     = 5e-5
save_interval     = 5000
```

`batch_size=128, num_train_steps=20000` 是当前主设置。它比总超参文件里的 batch 256 full recipe 更保守，适合作为这批小数据的第一阶段。不要在同一次训练中途改 batch size；如果后续想比较 batch 256，需要另开独立 run 并从 `pi05_base` 重新训练。

当前 selected10 train split 默认只用 sparse waypoint-call points，大约是 1000 demos / 4643 waypoint samples。真正要对齐的是 total samples seen / effective epochs：

```text
effective_epochs = num_train_steps * global_batch_size / 4643
```

主设置 `batch_size=128, num_train_steps=20000` 对应大约 2.56M waypoint samples，也就是约 551 effective epochs。若 validation 仍明显上涨，再 resume 到 40000 steps；如果 40000 仍明显上涨，才继续到 60000。若改 batch size，训练步数应按 batch 反向缩放：

| global batch | train steps | max steps | warmup steps | save interval | effective epochs |
|---:|---:|---:|---:|---:|---:|
| 128 | 20000 | 40000 | 10000 | 5000 | 551 |
| 128 | 40000 | 60000 | 10000 | 5000 | 1103 |
| 128 | 60000 | 60000 | 10000 | 5000 | 1654 |
| 256 | 10000 | 20000 | 10000 | 5000 | 551 |
| 64 | 40000 | 80000 | 10000 | 5000 | 551 |

主实验先用 global `128` 跑到 `20000` steps 并做 validation。若 validation 在 `20000` 仍明显上涨，再继续到 `40000`；如果 `40000` 仍明显上涨，才继续到 `60000`。

如果只做 LeRobot conversion/smoke，不需要启动 CoppeliaSim。online eval 后续才需要 RLBench/PyRep/CoppeliaSim。

参数依据：

- 本项目总超参文件：`/raid/home/than/zhiyuan/rlbench_baseline_hyperparams.md`
- OpenPI `pi05_droid_finetune`: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- OpenPI `pi05_libero`: https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py
- OpenPI training batch divisibility check: https://github.com/Physical-Intelligence/openpi/blob/main/scripts/train.py
- LeRobot pi0.5 finetune command: https://huggingface.co/docs/lerobot/en/pi05
- LeRobot pi0.5 memory discussion: https://github.com/huggingface/lerobot/issues/2216

## 安装

在 HPC 上 clone 这个 repo 后进入目录：

```bash
cd rlbench_pi05_waypoint_baseline_20260606
pip install -e . --no-deps
pip install -r requirements.txt
```

安装 OpenPI 到 repo 外部/内部指定目录：

```bash
PI05_ROOT=/path/to/pi05_baseline \
bash scripts/install_openpi_on_hpc.sh
```

默认会：

1. shallow clone `Physical-Intelligence/openpi`
2. 使用 `uv sync` 创建 OpenPI `.venv`
3. 不下载 git-lfs 大文件
4. 把 RLBench waypoint policy transform 复制进 OpenPI
5. 给 OpenPI `config.py` 添加 `pi05_rlbench_waypoint_h1`

这个 OpenPI config 的默认训练参数是：

```text
batch_size=128
num_train_steps=20000
warmup_steps=10000
save_interval=5000
fsdp_devices=8
```

如果 OpenPI 已经安装好了，只需要 patch：

```bash
OPENPI_DIR=/path/to/openpi \
bash scripts/patch_openpi_rlbench_config.sh
```

常用 cache root：

```bash
source scripts/setup_env.sh
echo "$OPENPI_DIR"
echo "$HF_LEROBOT_HOME"
echo "$OPENPI_DATA_HOME"
```

## 转换 LeRobot 数据

先做一个很小的 selected10 smoke conversion：

```bash
PI05_ROOT=/path/to/pi05_baseline \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
MAX_EPISODES_PER_TASK=2 \
SPLIT=train \
bash scripts/convert_selected10_waypoints_to_lerobot.sh
```

正式转换 train split：

```bash
PI05_ROOT=/path/to/pi05_baseline \
RGB_ROOT_200=/path/to/rgb_root_200 \
RGB_ROOT_400=/path/to/rgb_root_400 \
LOWDIM_ROOT_200=/path/to/lowdim_root_200 \
LOWDIM_ROOT_400=/path/to/lowdim_root_400 \
SPLIT=train \
bash scripts/convert_selected10_waypoints_to_lerobot.sh
```

默认输出到：

```text
${HF_LEROBOT_HOME}/rlbench/selected10_pi05_waypoint_h1
```

默认 `SAMPLE_EVERY_N=0`，也就是只使用 waypoint-call points。若想额外加入 segment 内的 replan-like samples，可以显式设置：

```bash
SAMPLE_EVERY_N=10 bash scripts/convert_selected10_waypoints_to_lerobot.sh
```

这会增加 planner 轨迹中间 observation，但 action 仍然是下一个 waypoint target。主 baseline 建议先保持 `SAMPLE_EVERY_N=0`。

## 数据 Smoke Test

转换后检查 LeRobot dataset 能否打开：

```bash
PI05_ROOT=/path/to/pi05_baseline \
bash scripts/smoke_lerobot_dataset.sh
```

这个命令只会读取少量 frame，打印 feature shape、dtype 和 task text。它不会启动训练，也不会下载 pi0.5 checkpoint。

## Norm Stats

OpenPI 训练前需要计算 `state` 和 `actions` normalization stats：

```bash
PI05_ROOT=/path/to/pi05_baseline \
bash scripts/compute_norm_stats.sh
```

小数据 smoke 可以限制帧数：

```bash
PI05_ROOT=/path/to/pi05_baseline \
bash scripts/compute_norm_stats.sh --max-frames 256
```

## 训练

训练入口：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1 \
bash scripts/train_pi05_waypoint_h1.sh
```

常用覆盖参数直接透传给 OpenPI `scripts/train.py`：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1_smoke \
WANDB_ENABLED=0 \
OVERWRITE=1 \
bash scripts/train_pi05_waypoint_h1.sh \
  --batch-size 2 \
  --num-train-steps 10 \
  --save-interval 10 \
  --fsdp-devices 1
```

8x A100 40GB 的 selected10 第一阶段 full run：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1 \
bash scripts/train_pi05_waypoint_h1.sh \
  --batch-size 128 \
  --num-train-steps 20000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --lr-schedule.warmup-steps 10000 \
  --fsdp-devices 8
```

如果 `5000/10000/15000/20000` 的 validation 仍在明显上涨，resume 同一个 run 到 40000 steps：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1 \
bash scripts/train_pi05_waypoint_h1.sh \
  --resume \
  --batch-size 128 \
  --num-train-steps 40000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --lr-schedule.warmup-steps 10000 \
  --fsdp-devices 8
```

如果 `40000` 仍明显最好，最后再 resume 到 60000 steps：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1 \
bash scripts/train_pi05_waypoint_h1.sh \
  --resume \
  --batch-size 128 \
  --num-train-steps 60000 \
  --save-interval 5000 \
  --keep-period 5000 \
  --lr-schedule.warmup-steps 10000 \
  --fsdp-devices 8
```

可选：在正式 full run 前，用很短的 smoke run 只检查显存、吞吐和 checkpoint 写入是否正常。smoke run 不是最终报告用的主训练：

```bash
PI05_ROOT=/path/to/pi05_baseline \
EXP_NAME=selected10_pi05_waypoint_h1_smoke \
WANDB_ENABLED=0 \
OVERWRITE=1 \
bash scripts/train_pi05_waypoint_h1.sh \
  --batch-size 128 \
  --num-train-steps 100 \
  --save-interval 100 \
  --lr-schedule.warmup-steps 100 \
  --fsdp-devices 8
```

第一阶段候选 checkpoint 建议完整跑 validation：

```text
5000
10000
15000
20000
```

如果 `20000` step 的 validation success 仍然最好，再继续到 `25000/30000/35000/40000`。如果 `40000` 仍然最好，再继续到 `45000/50000/55000/60000`。

当前 OpenPI config 使用：

- `config_name=pi05_rlbench_waypoint_h1`
- `pi0_config.Pi0Config(pi05=True, action_horizon=1, discrete_state_input=False)`
- model action dim 保持 pi0.5 默认 32，RLBench action 由 transform pad/trim 到前 7 维
- pretrained init: `gs://openpi-assets/checkpoints/pi05_base/params`
- selected10 first-stage default: global `batch_size=128`, `num_train_steps=20000`, `save_interval=5000`, `fsdp_devices=8`

## Online Eval

本 repo 当前重点是 HPC training/data conversion scaffold。online eval wrapper 后续应接到现有 RLBench eval 逻辑：

```text
live obs + full task instruction -> pi0.5 server -> absolute_rpy7 waypoint -> RLBench planner
```

这里不使用 event boundary，也不使用 Wan/video tokens。eval episode 选择应和 DP baseline manifest 的 val/test split 对齐。

## 注意事项

- 这是 full-task waypoint baseline，不是 event-oracle baseline。
- 每次只预测下一个 waypoint，`action_horizon=1`。
- 不用 selected10 video finetune 数据中的 `residual_video_image_paths`、subgoal image、event text 或 Wan hidden cache。
- 不使用 dense raw demo frames 作为默认监督；默认只使用 `0 + full_task_heuristic_waypoints`。
- LeRobot episode 长度可以不同；OpenPI dataloader 按 frame 采样，batch 内每个 sample shape 固定。
- `source_bundle` 决定使用哪个 RGB root 和 low-dim root，所以 HPC 上要同时配置 `RGB_ROOT_200/RGB_ROOT_400` 与 `LOWDIM_ROOT_200/LOWDIM_ROOT_400`。
- checkpoint、OpenPI repo、cache、LeRobot dataset 都不应该上传进这个 repo。
