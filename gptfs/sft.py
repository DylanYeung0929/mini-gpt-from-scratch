"""Supervised fine-tuning (instruction tuning) with optional LoRA."""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import GPTConfig
from .lora import apply_lora, lora_state_dict, merge_lora
from .model import GPT
from .tokenizer import TokenizerWrapper
from .utils import CSVMetricsLogger, autocast_ctx, get_device, human_int, save_json, set_seed


def load_instructions(path):
    """Supports Alpaca-style jsonl and OpenAI messages-style jsonl."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "messages" in obj:
                msgs = obj["messages"]
                user = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                assistant = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")
            else:
                user = obj.get("instruction", "")
                if obj.get("input"):
                    user += "\n" + obj["input"]
                assistant = obj.get("output", "")
            if user and assistant:
                rows.append({"user": user, "assistant": assistant})
    return rows


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, block_size):
        self.rows = rows
        self.tok = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        prompt_ids = self.tok.encode(f"<|user|>\n{r['user']}\n<|assistant|>\n")
        answer_ids = self.tok.encode(r["assistant"]) + [self.tok.eos_id]
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        # 模型内部不做 shift（预训练亦在外部构造 x/y），这里把 x=ids[:-1]、y=labels[1:]
        if len(ids) > self.block_size + 1:
            ids = ids[: self.block_size + 1]
            labels = labels[: self.block_size + 1]
        return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(labels[1:], dtype=torch.long)


def collate(batch, pad_id):
    max_len = max(len(x) for x, _ in batch)
    xs, ys = [], []
    for x, y in batch:
        pad = max_len - len(x)
        xs.append(F.pad(x, (0, pad), value=pad_id))
        ys.append(F.pad(y, (0, pad), value=-100))
    return torch.stack(xs), torch.stack(ys)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="X:/llm-course/checkpoints/pretrain/best.pt")
    p.add_argument("--tokenizer_dir", default="X:/llm-course/repo/artifacts/tokenizer")
    p.add_argument("--train_jsonl", default="X:/llm-course/data/raw/sft/train.jsonl")
    p.add_argument("--val_jsonl", default=None)
    p.add_argument("--out_dir", default="X:/llm-course/checkpoints/sft")
    p.add_argument("--runs_dir", default="X:/llm-course/runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=6)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--block_size", type=int, default=768)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--full_finetune", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--log_steps", action="store_true",
                   help="write per-step loss/lr/grad-norm/tokens to run_dir")
    p.add_argument("--save_merged", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    tok = TokenizerWrapper(args.tokenizer_dir)

    # ---- data
    rows = load_instructions(args.train_jsonl)
    random.Random(args.seed).shuffle(rows)
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_rows = load_instructions(args.val_jsonl)
    else:
        n_val = max(32, int(0.02 * len(rows)))
        val_rows, rows = rows[:n_val], rows[n_val:]
    print(f"SFT rows: train={len(rows)} val={len(val_rows)}")

    train_ds = SFTDataset(rows, tok, args.block_size)
    val_ds = SFTDataset(val_rows, tok, args.block_size)

    # ---- model
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    mcfg = GPTConfig(**ck["config"])
    mcfg.block_size = max(mcfg.block_size, args.block_size)
    mcfg.dropout = 0.0
    model = GPT(mcfg)
    model.load_state_dict(ck["model"])

    if args.full_finetune:
        trainable = sum(p.numel() for p in model.parameters())
        print(f"full fine-tuning, trainable params: {human_int(trainable)}")
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        n_lora = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                            dropout=args.lora_dropout)
        params = [p for p in model.parameters() if p.requires_grad]
        print(f"LoRA r={args.lora_r} alpha={args.lora_alpha}, "
              f"trainable params: {human_int(n_lora)} "
              f"({100*n_lora/sum(p.numel() for p in model.parameters()):.2f}%)")

    model.to(device)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))
    ctx = autocast_ctx(args.dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "float16"))

    run_name = args.run_name or time.strftime("sft-%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.runs_dir, run_name)
    logger = CSVMetricsLogger(run_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    save_json(os.path.join(args.out_dir, "sft_config.json"),
              {**vars(args), "model_config": mcfg.__dict__})

    steps_per_epoch = math.ceil(len(train_ds) / (args.batch_size * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = min(100, total_steps // 10 + 1)

    steps_path = os.path.join(run_dir, "steps.jsonl")
    if args.log_steps:
        open(steps_path, "w").close()  # fresh log per run
        gpu_props = (torch.cuda.get_device_properties(0)
                     if torch.cuda.is_available() else None)
        save_json(os.path.join(run_dir, "run_meta.json"), {
            "argv": sys.argv, "torch": torch.__version__, "device": str(device),
            "gpu": gpu_props.name if gpu_props else "cpu",
            "gpu_total_mem_mb": (round(gpu_props.total_memory / 2**20, 1)
                                 if gpu_props else None),
            "train_rows": len(train_ds), "val_rows": len(val_ds),
            "trainable_params": int(sum(p.numel() for p in params)),
            "total_steps": total_steps, "warmup": warmup, "dtype": args.dtype,
        })

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        ratio = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 * args.lr + 0.5 * (args.lr - 0.05 * args.lr) * (1 + math.cos(math.pi * ratio))

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(val_ds), args.batch_size):
            batch = [val_ds[j] for j in range(i, min(i + args.batch_size, len(val_ds)))]
            x, y = collate(batch, tok.eos_id)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss, _ = model(x, y)
            tot += loss.item()
            n += 1
        model.train()
        return tot / max(1, n)

    step = 0
    sup_seen = 0
    sup_acc = 0
    step_loss = 0.0
    model.train()
    t0 = time.time()
    for epoch in range(args.epochs):
        step_loss, sup_acc = 0.0, 0
        order = list(range(len(train_ds)))
        random.shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for it in range(0, len(order), args.batch_size):
            batch = [train_ds[j] for j in order[it: it + args.batch_size]]
            if not batch:
                continue
            x, y = collate(batch, tok.eos_id)
            x, y = x.to(device), y.to(device)
            with ctx:
                _, loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
            accum_loss += loss.item()
            step_loss += loss.item()
            sup_acc += int((y != -100).sum())

            if (it // args.batch_size + 1) % args.grad_accum == 0:
                lr = lr_at(step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                sup_seen += sup_acc
                if args.log_steps:
                    rec = {
                        "step": step, "epoch": epoch,
                        "loss": round(step_loss / args.grad_accum, 4),
                        "lr": lr, "grad_norm": round(gn, 4),
                        "supervised_tokens": sup_acc,
                        "time_s": round(time.time() - t0, 2),
                        "max_mem_alloc_mb": (round(torch.cuda.max_memory_allocated() / 2**20, 1)
                                             if torch.cuda.is_available() else None),
                    }
                    with open(steps_path, "a", encoding="utf-8") as sf:
                        sf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sup_acc = 0
                step_loss = 0.0
                if step % 10 == 0:
                    msg = logger.log(step=step, loss=round(accum_loss / (10 * args.grad_accum), 4),
                                     lr=f"{lr:.2e}", time_s=round(time.time() - t0, 1))
                    print(f"epoch {epoch} {msg}", flush=True)
                    accum_loss = 0.0
        if val_rows:
            vl = eval_loss()
            print(f"epoch {epoch} val_loss={vl:.4f} ppl={math.exp(vl):.2f}", flush=True)
            logger.log(step=step, val_loss=round(vl, 4), val_ppl=round(math.exp(vl), 3))

        if not args.full_finetune:
            torch.save({"lora": lora_state_dict(model), "config": mcfg.__dict__,
                        "lora_cfg": {"r": args.lora_r, "alpha": args.lora_alpha}},
                       os.path.join(args.out_dir, "adapter.pt"))

    # save final
    if args.full_finetune:
        torch.save({"model": model.state_dict(), "config": mcfg.__dict__, "sft": True},
                   os.path.join(args.out_dir, "sft_model.pt"))
    else:
        torch.save({"lora": lora_state_dict(model), "config": mcfg.__dict__,
                    "lora_cfg": {"r": args.lora_r, "alpha": args.lora_alpha}},
                   os.path.join(args.out_dir, "adapter.pt"))
        merged = merge_lora(model)
        torch.save({"model": merged.state_dict(), "config": mcfg.__dict__, "sft": True},
                   os.path.join(args.out_dir, "sft_merged.pt"))
    if args.log_steps:
        mem = {}
        if torch.cuda.is_available():
            mem = {
                "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                "max_memory_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            }
        save_json(os.path.join(run_dir, "run_summary.json"), {
            "steps": step, "epochs": args.epochs,
            "elapsed_s": round(time.time() - t0, 1),
            "supervised_tokens_seen": sup_seen,
            "rows_train": len(train_ds), "rows_val": len(val_ds), **mem})
    print(f"done in {time.time()-t0:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
