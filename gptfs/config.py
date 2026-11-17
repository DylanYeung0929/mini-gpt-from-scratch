"""Model / training configuration dataclasses."""
from dataclasses import dataclass, asdict
import json


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    block_size: int = 512          # context length
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12            # = n_head -> MHA, < n_head -> GQA
    n_embd: int = 768
    ffn_mult: int = 3              # SwiGLU: hidden = ffn_mult * n_embd (rounded to 64)
    dropout: float = 0.0
    bias: bool = False             # linear bias
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_weights: bool = True

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class TrainConfig:
    # data
    data_dir: str = "X:/llm-course/data/tokenized"
    out_dir: str = "X:/llm-course/checkpoints/pretrain"
    tokenizer_dir: str = "X:/llm-course/repo/artifacts/tokenizer"

    # optimization
    batch_size: int = 8
    grad_accum: int = 8
    max_steps: int = 20000
    warmup_steps: int = 500
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # runtime
    dtype: str = "bfloat16"        # bfloat16 | float16 | float32
    grad_checkpointing: bool = False
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1000      # rolling "last.pt"
    sample_interval: int = 1000
    seed: int = 1337
    num_workers: int = 0
    compile: bool = False
