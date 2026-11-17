"""Evaluate a checkpoint: validation loss / perplexity and sample generations."""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from gptfs.data import get_batch
from gptfs.generate import build_chat_prompt
from gptfs.inference import load_model, stream_generate
from gptfs.utils import get_device, human_int


STORY_PROMPTS = [
    "Once upon a time",
    "Tom was a little boy who",
    "The dragon looked at the moon and",
]

INSTRUCTION_PROMPTS = [
    "用一句话解释什么是机器学习。",
    "写一句关于春天的诗。",
    "Translate 'I love machine learning' into Chinese.",
    "Give three tips for learning English.",
    "What is the capital of France?",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="X:/llm-course/checkpoints/pretrain/best.pt")
    p.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer")
    p.add_argument("--data_dir", default="X:/llm-course/data/tokenized")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--sft", action="store_true", help="use chat prompt template")
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)
    print(f"loaded {args.ckpt} | params={human_int(model.num_params())} | device={device}")
    out_dir = args.out_dir or os.path.join(
        "X:/llm-course/runs", "eval-" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)

    result = {"checkpoint": args.ckpt, "params": model.num_params()}

    val_path = os.path.join(args.data_dir, "val.bin")
    if os.path.exists(val_path):
        data = np.memmap(val_path, dtype=np.uint16, mode="r")
        model.eval()
        losses = []
        with torch.no_grad():
            for _ in range(args.iters):
                x, y = get_batch(data, args.block_size, args.batch_size, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss, _ = model(x, y)
                losses.append(loss.item())
        val_loss = float(np.mean(losses))
        result["val_loss"] = val_loss
        result["val_ppl"] = math.exp(val_loss)
        print(f"val_loss={val_loss:.4f} ppl={math.exp(val_loss):.2f}")

    prompts = INSTRUCTION_PROMPTS if args.sft else STORY_PROMPTS
    lines = []
    for q in prompts:
        prompt = build_chat_prompt(q) if args.sft else q
        text = ""
        for text in stream_generate(model, tok, prompt,
                                    max_new_tokens=args.max_new_tokens,
                                    temperature=args.temperature, top_p=args.top_p,
                                    device=device):
            pass
        lines.append(f"### Prompt\n{q}\n\n### Response\n{text}\n")
        print(f"--- {q}\n{text}\n")
    with open(os.path.join(out_dir, "generations.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(out_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
