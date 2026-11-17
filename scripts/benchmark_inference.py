"""Measure single-request inference latency, throughput and peak VRAM (bf16)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from gptfs.generate import build_chat_prompt
from gptfs.inference import load_model
from gptfs.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="X:/llm-course/checkpoints/sft/sft_merged.pt")
    p.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer")
    p.add_argument("--lengths", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--sft", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)
    model = model.to(torch.bfloat16)

    prompt = (build_chat_prompt("用一句话解释什么是机器学习。")
              if args.sft else "Once upon a time")
    ids = tok.encode(prompt)

    rows = []
    for n in args.lengths:
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        model.generate(idx.clone(), 8, temperature=0.8, top_k=50, top_p=0.95)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = model.generate(idx.clone(), n, temperature=0.8, top_k=50, top_p=0.95,
                             repetition_penalty=1.1, eos_id=None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out.shape[1] - idx.shape[1]
        peak = (torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda" else float("nan"))
        rows.append({"max_new_tokens": n, "generated": new,
                     "total_ms": round(dt * 1000, 1),
                     "ms_per_token": round(dt * 1000 / new, 2),
                     "tokens_per_s": round(new / dt, 1),
                     "vram_peak_gb": round(peak, 2)})
        print(rows[-1], flush=True)

    out_path = args.out or os.path.join(
        "X:/llm-course/runs", "bench-" + time.strftime("%Y%m%d-%H%M%S") + ".json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"ckpt": args.ckpt, "dtype": "bfloat16", "prompt": prompt,
                   "results": rows}, f, ensure_ascii=False, indent=2)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
