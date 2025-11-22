#!/usr/bin/env python
"""
Run ACT policy inference on a single recorded observation and capture latency
metrics with torch_npu.profiler.

Example:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    conda activate op
    python scripts/profile_act_minimal.py \
        --policy-path /home/HwHiAiUser/dora_ws/act/pretrained_model \
        --dataset /home/HwHiAiUser/DoRobot/dataset/20251108/experimental/so101-1108/ \
        --iters 10 --profile
"""

from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from operating_platform.config.policies import PreTrainedConfig
from operating_platform.dataset.dorobot_dataset import DoRobotDataset
from operating_platform.policy.factory import make_policy
from operating_platform.utils.dataset import DEFAULT_FEATURES, build_dataset_frame
from operating_platform.utils.utils import get_safe_torch_device

try:
    from torch_npu.profiler import ProfilerActivity, profile
except ImportError:  # pragma: no cover - optional dependency
    profile = None
    ProfilerActivity = None


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def prepare_observation(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Mimic inference.predict_action preprocessing for a single frame."""
    obs: dict[str, torch.Tensor] = {}
    for name, array in batch.items():
        tensor = torch.from_numpy(_to_numpy(array))
        if "image" in name:
            if tensor.dim() == 3 and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).contiguous()
            tensor = tensor.to(dtype=torch.float32) / 255.0
        else:
            tensor = tensor.to(dtype=torch.float32)
        obs[name] = tensor.unsqueeze(0).to(device)
    obs["task"] = ""
    obs["robot_type"] = ""
    return obs


def build_frame_with_fallback(
    ds_features: dict[str, dict],
    values: dict[str, Any],
    prefix: str,
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    for key, ft in ds_features.items():
        if key in DEFAULT_FEATURES or not key.startswith(prefix):
            continue
        if ft["dtype"] == "float32" and len(ft["shape"]) == 1:
            arr = []
            for name in ft["names"]:
                val = values.get(name, 0.0)
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().numpy()
                if isinstance(val, (list, tuple, np.ndarray)):
                    val = np.asarray(val).reshape(-1)[0]
                arr.append(float(val))
            frame[key] = np.array(arr, dtype=np.float32)
        elif ft["dtype"] in {"image", "video"}:
            src_key = key.removeprefix(f"{prefix}.images.")
            if src_key in values:
                frame[key] = values[src_key]
    return frame


def run_inference(
    policy_path: Path,
    dataset_path: Path,
    sample_index: int,
    iters: int,
    warmup: int,
    use_profiler: bool,
) -> None:
    dataset = DoRobotDataset(repo_id=str(dataset_path), root=None, download_videos=True)
    sample = dataset[sample_index]
    try:
        observation_frame = build_dataset_frame(dataset.features, sample, prefix="observation")
    except KeyError as exc:
        print(f"[WARN] Missing key {exc}. Falling back to zero-filled state vector.")
        observation_frame = build_frame_with_fallback(dataset.features, sample, prefix="observation")

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy = make_policy(cfg, ds_meta=dataset.meta)
    device = get_safe_torch_device(policy.config.device, log=True)
    policy.to(device)
    policy.eval()

    obs = prepare_observation(observation_frame, device)

    @torch.no_grad()
    def single_step() -> None:
        _ = policy.select_action(obs)

    if device.type == "npu":
        torch.npu.synchronize()
    for _ in range(warmup):
        single_step()
    if device.type == "npu":
        torch.npu.synchronize()

    ctx = (
        profile(
            activities=[ProfilerActivity.NPU],
            record_shapes=True,
            start_npu_trace=True,
            with_stack=True,
        )
        if use_profiler and profile is not None
        else nullcontext()
    )

    start = time.perf_counter()
    with ctx as prof:
        for _ in range(iters):
            single_step()
        if prof is not None and device.type == "npu":
            torch.npu.synchronize()
    elapsed = (time.perf_counter() - start) / max(iters, 1)

    if use_profiler and profile is not None:
        print(prof.key_averages().table(sort_by="self_npu_time_total", row_limit=20))

    print(f"Average latency over {iters} iters: {elapsed*1000:.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile ACT inference on a single recorded frame.")
    parser.add_argument("--policy-path", type=Path, required=True, help="Path to pretrained ACT model directory.")
    parser.add_argument("--dataset", type=Path, required=True, help="Local DoRobot dataset directory.")
    parser.add_argument("--sample-index", type=int, default=0, help="Frame index to profile.")
    parser.add_argument("--iters", type=int, default=5, help="Number of profiled iterations.")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations before measuring.")
    parser.add_argument("--profile", action="store_true", help="Enable torch_npu.profiler output.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        policy_path=args.policy_path,
        dataset_path=args.dataset,
        sample_index=args.sample_index,
        iters=args.iters,
        warmup=args.warmup,
        use_profiler=args.profile,
    )
