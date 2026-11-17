"""Shared helpers: seeding, LR schedule, logging."""
import math
import os
import time

import torch


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lr(step: int, cfg):
    """Linear warmup followed by cosine decay to min_lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if step >= cfg.max_steps:
        return cfg.min_lr
    ratio = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * ratio))


class CSVMetricsLogger:
    """Append scalar metrics to runs/<run>/log.csv and a human-readable lines file."""

    FIELDS = ["step", "loss", "val_loss", "val_ppl", "lr", "tok_s", "tokens", "time_s"]

    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.csv_path = os.path.join(run_dir, "log.csv")
        self.lines_path = os.path.join(run_dir, "train.log")
        self._header_written = os.path.exists(self.csv_path)
        self._fields = list(self.FIELDS)
        self._t0 = time.time()

    def log(self, **metrics):
        metrics.setdefault("time_s", time.time() - self._t0)
        with open(self.csv_path, "a", encoding="utf-8") as f:
            if not self._header_written:
                f.write(",".join(self._fields) + "\n")
                self._header_written = True
            f.write(",".join(str(metrics.get(k, "")) for k in self._fields) + "\n")
        msg = " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in metrics.items())
        with open(self.lines_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        return msg


def human_int(n):
    for unit in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(int(n))
        n /= 1000.0
    return f"{n:.1f}P"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def autocast_ctx(dtype: str):
    from contextlib import nullcontext
    if dtype == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def save_json(path, obj):
    import json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
