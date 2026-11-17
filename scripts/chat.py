"""Minimal terminal chat REPL (streaming)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gptfs.generate import build_chat_prompt
from gptfs.inference import load_model, stream_generate
from gptfs.utils import get_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="X:/llm-course/checkpoints/sft/sft_merged.pt")
    p.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    args = p.parse_args()

    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)
    print("输入问题，输入 /exit 退出。")
    while True:
        try:
            q = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q in {"exit", "/exit", "quit"}:
            break
        print("模型> ", end="", flush=True)
        prompt = build_chat_prompt(q)
        last = ""
        for piece in stream_generate(model, tok, prompt,
                                     max_new_tokens=args.max_new_tokens,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     repetition_penalty=1.1, device=device):
            print(piece[len(last):], end="", flush=True)
            last = piece
        print()


if __name__ == "__main__":
    main()
