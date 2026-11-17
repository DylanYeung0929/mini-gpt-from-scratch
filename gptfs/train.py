"""Pretraining loop (custom, minimal-dependency)."""
import argparse
import os
import time

import numpy as np
import torch

from .config import GPTConfig, TrainConfig
from .data import get_batch
from .model import GPT
from .utils import (CSVMetricsLogger, autocast_ctx, get_device, get_lr,
                    human_int, save_json, set_seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="X:/llm-course/data/tokenized")
    p.add_argument("--out_dir", default="X:/llm-course/checkpoints/pretrain")
    p.add_argument("--runs_dir", default="X:/llm-course/runs")
    p.add_argument("--run_name", default=None)
    p.add_argument("--tokenizer_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--block_size", type=int, default=None)
    p.add_argument("--n_layer", type=int, default=None)
    p.add_argument("--n_embd", type=int, default=None)
    p.add_argument("--n_head", type=int, default=None)
    p.add_argument("--n_kv_head", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--grad_checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def build_configs(args):
    tcfg = TrainConfig()
    for k in ["data_dir", "out_dir", "runs_dir", "tokenizer_dir", "max_steps",
              "batch_size", "grad_accum", "lr", "dtype"]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(tcfg, k, v)
    mcfg = GPTConfig()
    for k in ["block_size", "n_layer", "n_embd", "n_head", "n_kv_head"]:
        v = getattr(args, k, None)
        if v is not None:
            setattr(mcfg, k, v)
    return mcfg, tcfg


@torch.no_grad()
def estimate_loss(model, val_data, block_size, batch_size, device, iters, ctx):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(val_data, block_size, batch_size, device)
        with ctx:
            _, loss, _ = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    args = parse_args()
    mcfg, tcfg = build_configs(args)
    if args.grad_checkpointing:
        tcfg.grad_checkpointing = True
    tcfg.compile = args.compile
    set_seed(tcfg.seed)
    device = get_device()
    print(f"device={device} | params(est) | dtype={tcfg.dtype}")

    run_name = args.run_name or time.strftime("pretrain-%Y%m%d-%H%M%S")
    run_dir = os.path.join(tcfg.runs_dir, run_name)
    logger = CSVMetricsLogger(run_dir)
    os.makedirs(tcfg.out_dir, exist_ok=True)

    # ---- data
    train_path = os.path.join(tcfg.data_dir, "train.bin")
    val_path = os.path.join(tcfg.data_dir, "val.bin")
    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint16, mode="r")
    print(f"tokens: train={human_int(len(train_data))} val={human_int(len(val_data))}")

    # 词表大小以数据目录里的 dataset_meta.json 为准（支持换分词器），
    # 没有时回退到默认分词器的 meta。
    tok_meta = os.path.join(tcfg.data_dir, "dataset_meta.json")
    if not os.path.exists(tok_meta):
        tok_meta = "X:/llm-course/repo/artifacts/tokenizer/tokenizer_meta.json"
    if os.path.exists(tok_meta):
        import json
        with open(tok_meta, encoding="utf-8") as f:
            meta = json.load(f)
        mcfg.vocab_size = meta.get("tokenizer_vocab_size", meta.get("vocab_size"))
    print(f"vocab_size={mcfg.vocab_size}")

    # ---- model / optim
    model = GPT(mcfg).to(device)
    model.grad_checkpointing = tcfg.grad_checkpointing
    print(f"model params: {human_int(model.num_params())} "
          f"(non-embedding {human_int(model.num_params(non_embedding=True))})")
    if tcfg.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile enabled")
        except Exception as e:
            print(f"compile failed, continuing eager: {e}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg.lr, betas=(tcfg.beta1, tcfg.beta2),
        weight_decay=tcfg.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(tcfg.dtype == "float16"))
    ctx = autocast_ctx(tcfg.dtype)

    step = 0
    best_val = float("inf")
    tokens_seen = 0
    last_ckpt = os.path.join(tcfg.out_dir, "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        step = ck["step"] + 1
        tokens_seen = ck.get("tokens_seen", 0)
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from step {step}")

    save_json(os.path.join(tcfg.out_dir, "model_config.json"),
              {**mcfg.__dict__, "train_config": tcfg.__dict__})

    sample_prompts = ["Once upon a time", "Tom and his friend", "从前，有一个小女孩"]
    tokens_per_step = tcfg.batch_size * tcfg.grad_accum * mcfg.block_size
    t_interval = time.time()
    model.train()

    for step in range(step, tcfg.max_steps):
        lr = get_lr(step, tcfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(tcfg.grad_accum):
            x, y = get_batch(train_data, mcfg.block_size, tcfg.batch_size, device)
            with ctx:
                _, loss, _ = model(x, y)
            loss = loss / tcfg.grad_accum
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if tcfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step
        if step % tcfg.log_interval == 0:
            dt = time.time() - t_interval
            t_interval = time.time()
            tok_s = tokens_per_step * tcfg.log_interval / max(dt, 1e-9)
            msg = logger.log(step=step, loss=round(loss_accum, 4), lr=f"{lr:.2e}",
                             tok_s=int(tok_s), tokens=human_int(tokens_seen))
            print(msg, flush=True)

        do_eval = (step % tcfg.eval_interval == 0) or (step + 1 == tcfg.max_steps)
        if do_eval:
            val_loss = estimate_loss(model, val_data, mcfg.block_size,
                                     tcfg.batch_size, device, tcfg.eval_iters, ctx)
            perplexity = float(np.exp(val_loss))
            print(f"eval step={step} val_loss={val_loss:.4f} ppl={perplexity:.2f}", flush=True)
            logger.log(step=step, val_loss=round(val_loss, 4), val_ppl=round(perplexity, 3))
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "step": step,
                            "val_loss": val_loss, "config": mcfg.__dict__},
                           os.path.join(tcfg.out_dir, "best.pt"))

        if step % tcfg.sample_interval == 0 and step > 0:
            from .generate import generate_text
            from .tokenizer import TokenizerWrapper
            try:
                tok = TokenizerWrapper(tcfg.tokenizer_dir)
                lines = [f"===== step {step} ====="]
                for p in sample_prompts:
                    out = generate_text(model, tok, p, max_new_tokens=120,
                                        temperature=0.8, top_k=50, device=device)
                    lines.append(f"[prompt] {p}\n{out}\n")
                with open(os.path.join(run_dir, "samples.txt"), "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                print(lines[1][:200], flush=True)
            except Exception as e:
                print(f"sample failed (tokenizer missing?): {e}")

        if step % tcfg.save_interval == 0 or step + 1 == tcfg.max_steps:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "tokens_seen": tokens_seen, "best_val": best_val,
                        "config": mcfg.__dict__}, last_ckpt)
            print(f"saved checkpoint at step {step}", flush=True)

    torch.save({"model": model.state_dict(), "step": tcfg.max_steps - 1,
                "tokens_seen": tokens_seen, "best_val": best_val, "config": mcfg.__dict__},
               os.path.join(tcfg.out_dir, "final.pt"))
    print(f"training done. best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
