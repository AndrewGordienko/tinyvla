"""Bounded real SmolVLA base-checkpoint forward/backward/save-reload smoke test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

from tinyvla.fast_dataset import FastChunkDataset
from tinyvla.runtime import load_runtime
from tinyvla.trainability import set_trainable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    timestamps = {"action": [i / meta.fps for i in range(SmolVLAConfig().chunk_size)]}
    ds = FastChunkDataset(args.repo_id, root=args.root, delta_timestamps=timestamps)
    raw = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)))
    runtime = load_runtime(args.model, meta=meta, dataset_root=args.root, device=device,
                           stats_source="dataset", base_checkpoint=True)
    policy = runtime.policy
    trainable = set_trainable(policy, "expert")
    batch = runtime.preprocessor(dict(raw))
    policy.train(); policy.zero_grad(set_to_none=True)
    torch.manual_seed(17)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), device=device)
    time = torch.full((1,), 0.37, device=device)
    loss, details = policy.forward(batch, noise=noise, time=time)
    loss.backward()
    grads = [p.grad.detach().float().norm().item() for p in policy.parameters() if p.grad is not None]
    finite = bool(torch.isfinite(loss).item() and all(torch.isfinite(torch.tensor(g)) for g in grads))
    policy.eval()
    with torch.inference_mode():
        before, _ = policy.forward(batch, noise=noise, time=time)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    state_path = out.with_suffix(".state.pt")
    torch.save(policy.state_dict(), state_path)
    reloaded = load_runtime(args.model, meta=meta, dataset_root=args.root, device=device,
                            stats_source="dataset", base_checkpoint=True).policy
    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=True), strict=True)
    reloaded.eval()
    with torch.inference_mode():
        after, _ = reloaded.forward(batch, noise=noise, time=time)
    result = {
        "model": args.model, "dataset": args.root, "device": str(device),
        "trainable_params": trainable, "loss": float(loss), "loss_details": details,
        "gradient_norm": float(sum(g * g for g in grads) ** 0.5),
        "finite_gradients": finite, "save_reload_loss_abs_diff": float((before - after).abs()),
        "save_reload_equivalent": bool(torch.allclose(before, after, atol=1e-6, rtol=1e-6)),
        "state_artifact": str(state_path),
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not finite or not result["save_reload_equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
