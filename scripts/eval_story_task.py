"""Held-out story-instruction evaluation before/after SFT.

Twelve Chinese and twelve English prompts (topic + two required characters /
settings) are generated with a fixed seed.  For each output we measure:

  * keyword coverage -- how many of the three required nouns appear;
  * language consistency -- CJK (or Latin) character ratio of the output;
  * repetition -- 1 - distinct-4-gram ratio over characters.

The subset is written for this evaluation only and is not part of the SFT
data.  By default pretrained checkpoints receive the bare instruction and SFT
checkpoints receive the training template; --prompt_format decouples the two
so that checkpoint and prompt format can be compared independently.

Usage:
  python scripts/eval_story_task.py --ckpt <pt> --tokenizer_dir <dir> \
      --mode pre|sft [--prompt_format auto|bare|template] --tag bi_pre --out <json>
"""
import argparse
import json
import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptfs.inference import load_model  # noqa: E402
from gptfs.utils import get_device, set_seed  # noqa: E402

PROMPTS = [
    ("zh", "写一个关于兔子的短故事。故事里要有兔子、妈妈和森林。", ["兔子", "妈妈", "森林"]),
    ("zh", "写一个关于狐狸的短故事。故事里要有狐狸、朋友和河边。", ["狐狸", "朋友", "河"]),
    ("zh", "写一个关于熊的短故事。故事里要有熊、蜂蜜和山洞。", ["熊", "蜂蜜", "山洞"]),
    ("zh", "写一个关于小鸟的短故事。故事里要有小鸟、大树和天空。", ["鸟", "树", "天空"]),
    ("zh", "写一个关于小狗的短故事。故事里要有小狗、球和公园。", ["狗", "球", "公园"]),
    ("zh", "写一个关于小猫的短故事。故事里要有小猫、鱼和厨房。", ["猫", "鱼", "厨房"]),
    ("zh", "写一个关于鸭子的短故事。故事里要有鸭子、池塘和雨。", ["鸭", "池塘", "雨"]),
    ("zh", "写一个关于松鼠的短故事。故事里要有松鼠、松果和秋天。", ["松鼠", "松果", "秋天"]),
    ("zh", "写一个关于马的短故事。故事里要有马、草地和比赛。", ["马", "草地", "比赛"]),
    ("zh", "写一个关于羊的短故事。故事里要有羊、山坡和狼。", ["羊", "山坡", "狼"]),
    ("zh", "写一个关于猪的短故事。故事里要有猪、泥坑和农场。", ["猪", "泥坑", "农场"]),
    ("zh", "写一个关于刺猬的短故事。故事里要有刺猬、苹果和夜晚。", ["刺猬", "苹果", "夜晚"]),
    ("en", "Write a short story about a rabbit, its mother, and a forest.",
     ["rabbit", "mother", "forest"]),
    ("en", "Write a short story about a fox, a friend, and a river.",
     ["fox", "friend", "river"]),
    ("en", "Write a short story about a bear, honey, and a cave.",
     ["bear", "honey", "cave"]),
    ("en", "Write a short story about a bird, a big tree, and the sky.",
     ["bird", "tree", "sky"]),
    ("en", "Write a short story about a dog, a ball, and a park.",
     ["dog", "ball", "park"]),
    ("en", "Write a short story about a cat, a fish, and a kitchen.",
     ["cat", "fish", "kitchen"]),
    ("en", "Write a short story about a duck, a pond, and rain.",
     ["duck", "pond", "rain"]),
    ("en", "Write a short story about a squirrel, an acorn, and autumn.",
     ["squirrel", "acorn", "autumn"]),
    ("en", "Write a short story about a horse, grass, and a race.",
     ["horse", "grass", "race"]),
    ("en", "Write a short story about a sheep, a hill, and a wolf.",
     ["sheep", "hill", "wolf"]),
    ("en", "Write a short story about a pig, mud, and a farm.",
     ["pig", "mud", "farm"]),
    ("en", "Write a short story about a hedgehog, an apple, and the night.",
     ["hedgehog", "apple", "night"]),
]


def cjk_ratio(s):
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    latin = sum(1 for c in s if ("a" <= c.lower() <= "z"))
    return cjk / max(1, cjk + latin), latin / max(1, cjk + latin)


def distinct_ratio(s, n=4):
    s = re.sub(r"\s+", "", s)
    if len(s) <= n:
        return 1.0
    grams = [s[i:i + n] for i in range(len(s) - n + 1)]
    return len(set(grams)) / len(grams)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--mode", choices=["pre", "sft"], required=True)
    p.add_argument("--prompt_format", choices=["auto", "bare", "template"],
                   default="auto", help="auto: template for sft, bare for pre")
    p.add_argument("--tag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jsonl", default=None, help="optional per-sample JSONL")
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)
    model.eval()
    set_seed(args.seed)
    prompt_format = args.prompt_format
    if prompt_format == "auto":
        prompt_format = "template" if args.mode == "sft" else "bare"
    condition = f"{args.tag}:{prompt_format}"

    rows = []
    for i, (lang, prompt, keys) in enumerate(PROMPTS):
        if prompt_format == "template":
            text = f"<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            text = prompt + "\n"
        ids = tok.encode(text)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(idx, max_new_tokens=args.max_new_tokens,
                                 temperature=args.temperature, top_k=args.top_k,
                                 top_p=args.top_p,
                                 repetition_penalty=args.repetition_penalty,
                                 eos_id=tok.eos_id)
        latency_ms = (time.perf_counter() - t0) * 1000
        n_new = int(out.shape[1] - len(ids))
        gen = tok.decode(out[0, len(ids):].tolist()).replace("<|endoftext|>", "").strip()
        low = gen.lower()
        coverage = sum(1 for k in keys if k.lower() in low)
        zh_r, en_r = cjk_ratio(gen)
        lang_ratio = zh_r if lang == "zh" else en_r
        d4 = distinct_ratio(gen)
        fails = []
        if coverage < len(keys):
            fails.append("missing_keyword")
        if lang_ratio < 0.85:
            fails.append("language_mix")
        if d4 < 0.70:
            fails.append("repetition")
        rows.append({
            "run_id": args.tag, "sample_id": i, "condition": condition,
            "prompt_format": prompt_format, "ckpt": args.ckpt, "mode": args.mode,
            "seed": args.seed, "sampling": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature, "top_k": args.top_k,
                "top_p": args.top_p, "repetition_penalty": args.repetition_penalty},
            "lang": lang, "prompt": prompt, "keywords": keys, "output": gen,
            "coverage": coverage, "lang_ratio": round(lang_ratio, 3),
            "distinct4": round(d4, 3), "repetition": round(1 - d4, 3),
            "chars": len(gen), "output_tokens": n_new,
            "stop_reason": "eos" if int(out[0, -1]) == tok.eos_id else "max_new_tokens",
            "latency_ms": round(latency_ms, 1), "failures": fails,
            "success": bool(coverage == len(keys) and lang_ratio >= 0.85 and d4 >= 0.70),
        })

    def mean(xs):
        return round(sum(xs) / len(xs), 3)

    failure_counts = {}
    for r in rows:
        for f in r["failures"]:
            failure_counts[f] = failure_counts.get(f, 0) + 1

    summary = {
        "tag": args.tag, "condition": condition, "prompt_format": prompt_format,
        "ckpt": args.ckpt, "mode": args.mode, "seed": args.seed,
        "failure_counts": failure_counts,
        "n": len(rows),
        "coverage_mean": mean([r["coverage"] for r in rows]),
        "full_coverage_rate": mean([r["coverage"] == 3 for r in rows]),
        "lang_consistency_rate": mean([r["lang_ratio"] >= 0.85 for r in rows]),
        "distinct4_mean": mean([r["distinct4"] for r in rows]),
        "repetition_mean": mean([r["repetition"] for r in rows]),
        "success_rate": mean([r["success"] for r in rows]),
        "zh_coverage_mean": mean([r["coverage"] for r in rows if r["lang"] == "zh"]),
        "en_coverage_mean": mean([r["coverage"] for r in rows if r["lang"] == "en"]),
        "zh_success_rate": mean([r["success"] for r in rows if r["lang"] == "zh"]),
        "en_success_rate": mean([r["success"] for r in rows if r["lang"] == "en"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
