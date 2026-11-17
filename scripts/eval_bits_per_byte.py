"""Cross-tokenizer comparison on the SFT validation split.

Perplexities of models with different vocabularies are not comparable, because
the same text is cut into a different number of tokens.  This script therefore
reports the negative log-likelihood per *UTF-8 byte* (bits/char for ASCII,
bits/byte for mixed text) of the supervised answer text, which is invariant to
the tokenizer.

It rebuilds exactly the SFT validation split used in training (seed 1337,
first 2%), evaluates a checkpoint with the same template/loss mask, and
decodes the evaluated label tokens with that model's own tokenizer to count
the original bytes.

Usage:
  python scripts/eval_bits_per_byte.py --ckpt <pt> --tokenizer_dir <dir> \
      --train_jsonl <jsonl> --tag bi_sft --out results.json
"""
import argparse
import json
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gptfs.inference import load_model  # noqa: E402
from gptfs.sft import SFTDataset, collate, load_instructions  # noqa: E402
from gptfs.utils import get_device  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--filter_tokenizer_dir", default=None,
                   help="drop rows that overflow block_size under this second tokenizer, "
                        "so every checkpoint scores the identical set of texts")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def run(model, tok, rows, args, device):
    ds = SFTDataset(rows, tok, args.block_size)
    nll_sum, n_tok, n_bytes = 0.0, 0, 0
    for i in range(0, len(ds), args.batch_size):
        batch = [ds[j] for j in range(i, min(i + args.batch_size, len(ds)))]
        x, y = collate(batch, tok.eos_id)
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, _ = model(x)
            batch_sum = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        nll_sum += batch_sum.item()
        n_tok += (y != -100).sum().item()
        for row_y in y:
            ids = row_y[row_y != -100].tolist()
            text = tok.tok.decode(ids, skip_special_tokens=True)
            n_bytes += len(text.encode("utf-8"))
    return {
        "nll_nats": nll_sum,
        "n_tokens": n_tok,
        "n_bytes": n_bytes,
        "nats_per_token": nll_sum / max(1, n_tok),
        "ppl": math.exp(nll_sum / max(1, n_tok)),
        "bits_per_byte": nll_sum / math.log(2) / max(1, n_bytes),
    }


def main():
    args = parse_args()
    device = get_device()
    model, tok = load_model(args.ckpt, args.tokenizer_dir, device)
    model.eval()

    rows = load_instructions(args.train_jsonl)
    random.Random(args.seed).shuffle(rows)
    n_val = max(32, int(0.02 * len(rows)))
    val_rows = rows[:n_val]

    if args.filter_tokenizer_dir:
        from gptfs.tokenizer import TokenizerWrapper
        other = TokenizerWrapper(args.filter_tokenizer_dir)

        def fits(t, r):
            return (len(t.encode(f"<|user|>\n{r['user']}\n<|assistant|>\n"))
                    + len(t.encode(r["assistant"])) + 1) <= args.block_size + 1

        val_rows = [r for r in val_rows if fits(tok, r) and fits(other, r)]
        print(f"rows kept after truncation filter: {len(val_rows)}")

    def is_chinese(r):
        s = r["user"] + r["assistant"]
        cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        return cjk >= max(4, 0.15 * len(s))

    res = {"tag": args.tag, "ckpt": args.ckpt, "n": len(val_rows)}
    subsets = {
        "all": val_rows,
        "en": [r for r in val_rows if not is_chinese(r)],
        "zh": [r for r in val_rows if is_chinese(r)],
    }
    for name, subset in subsets.items():
        m = run(model, tok, subset, args, device)
        for k, v in m.items():
            res[f"{name}_{k}"] = v
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if args.out:
        prev = []
        if os.path.exists(args.out):
            with open(args.out, encoding="utf-8") as f:
                prev = json.load(f)
        prev = [r for r in prev if r.get("tag") != args.tag] + [res]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
