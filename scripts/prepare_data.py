"""Prepare tokenizer and uint16 token bins.

Examples (run from repo root):
  python -m scripts.prepare_data tokenizer --input X:/llm-course/data/raw/train.txt \
      --out X:/llm-course/repo/artifacts/tokenizer --vocab_size 32768
  python -m scripts.prepare_data bins \
      --train_text X:/llm-course/data/raw/TinyStoriesV2-GPT4-train.txt \
      --val_text X:/llm-course/data/raw/TinyStoriesV2-GPT4-valid.txt \
      --out_dir X:/llm-course/data/tokenized
"""
import argparse
import json
import os

from gptfs.data import prepare_token_bin
from gptfs.tokenizer import TokenizerWrapper, train_bpe


def cmd_tokenizer(args):
    files = [p.strip() for p in args.input.split(",") if p.strip()]
    train_bpe(files, args.out, vocab_size=args.vocab_size,
              sample_chars=args.sample_chars)
    tok = TokenizerWrapper(args.out)
    print(f"tokenizer trained: vocab={tok.vocab_size} -> {args.out}")


def cmd_bins(args):
    tok = TokenizerWrapper(args.tokenizer_dir)
    print(f"encoding train -> {args.out_dir}/train.bin")
    tr = prepare_token_bin(args.train_text, os.path.join(args.out_dir, "train.bin"), tok)
    print(f"train: {tr}")
    print(f"encoding val -> {args.out_dir}/val.bin")
    va = prepare_token_bin(args.val_text, os.path.join(args.out_dir, "val.bin"), tok)
    print(f"val: {va}")
    meta = {"tokenizer_vocab_size": tok.vocab_size, "train": tr, "val": va}
    with open(os.path.join(args.out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("tokenizer")
    pt.add_argument("--input", required=True, help="comma separated corpus files")
    pt.add_argument("--out", default="X:/llm-course/repo/artifacts/tokenizer")
    pt.add_argument("--vocab_size", type=int, default=32768)
    pt.add_argument("--sample_chars", type=int, default=300_000_000)
    pt.set_defaults(func=cmd_tokenizer)

    pb = sub.add_parser("bins")
    pb.add_argument("--train_text", required=True)
    pb.add_argument("--val_text", required=True)
    pb.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer")
    pb.add_argument("--out_dir", default="X:/llm-course/data/tokenized")
    pb.set_defaults(func=cmd_bins)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
