"""Measure one pretraining step's GPU memory, split by allocation type.

The training loop keeps fp32 weights and fp32 AdamW state and only runs the
forward/backward math under bfloat16 autocast.  This script therefore reports:

  * parameter / gradient / optimizer-state bytes by their real dtype;
  * peak CUDA *allocated* and *reserved* memory during one step;
  * device-level total and free memory (includes CUDA context, caches, etc.).

Usage:
  python scripts/measure_vram.py --model_config <json> [--batch_size 4] [--block_size 1024]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptfs.config import GPTConfig  # noqa: E402
from gptfs.model import GPT  # noqa: E402
from gptfs.utils import autocast_ctx  # noqa: E402


def mib(n):
    return round(n / 2**20, 1)


def state_bytes(opt):
    tot = 0
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", required=True)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--block_size", type=int, default=1024)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = "cuda"
    with open(args.model_config, encoding="utf-8") as f:
        raw = json.load(f)
    valid = {k: v for k, v in raw.items() if k in GPTConfig.__dataclass_fields__}
    cfg = GPTConfig(**valid)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95),
                            weight_decay=0.1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_alloc = torch.cuda.memory_allocated()

    param_bytes = sum(q.numel() * q.element_size() for q in model.parameters())

    x = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    y = torch.randint(0, cfg.vocab_size, (args.batch_size, args.block_size), device=device)
    ctx = autocast_ctx(args.dtype)

    with ctx:
        _, loss, _ = model(x, y)
    fwd_alloc = torch.cuda.memory_allocated()
    loss.backward()
    bwd_alloc = torch.cuda.memory_allocated()
    grad_bytes = sum(q.grad.numel() * q.grad.element_size()
                     for q in model.parameters() if q.grad is not None)
    opt.step()
    opt.zero_grad(set_to_none=True)
    after_alloc = torch.cuda.memory_allocated()
    opt_bytes = state_bytes(opt)

    free_, total = torch.cuda.mem_get_info()
    res = {
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "param_bytes": param_bytes,
        "param_MiB": mib(param_bytes),
        "grad_bytes": grad_bytes,
        "grad_MiB": mib(grad_bytes),
        "adam_state_bytes": opt_bytes,
        "adam_state_MiB": mib(opt_bytes),
        "base_alloc_MiB": mib(base_alloc),
        "forward_alloc_MiB": mib(fwd_alloc),
        "after_backward_alloc_MiB": mib(bwd_alloc),
        "after_step_alloc_MiB": mib(after_alloc),
        "peak_allocated_MiB": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_MiB": mib(torch.cuda.max_memory_reserved()),
        "device_total_MiB": mib(total),
        "device_free_MiB": mib(free_),
        "device_used_MiB": mib(total - free_),
    }
    print(json.dumps(res, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
