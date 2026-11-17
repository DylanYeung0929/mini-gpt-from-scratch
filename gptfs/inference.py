"""Checkpoint loading + streaming generation shared by CLI / web app."""
import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import TokenizerWrapper


def load_model(ckpt_path, tokenizer_dir, device):
    tok = TokenizerWrapper(tokenizer_dir)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ck.get("config") or ck.get("model_config")
    mcfg = GPTConfig(**cfg_dict)
    mcfg.vocab_size = tok.vocab_size
    model = GPT(mcfg)
    state = ck.get("model")
    if state is None:
        raise KeyError("checkpoint has no 'model' weights (use sft_merged.pt)")
    model.load_state_dict(state)
    model.eval().to(device)
    return model, tok


@torch.inference_mode()
def stream_generate(model, tok, prompt, max_new_tokens=256, temperature=0.8,
                    top_k=50, top_p=0.95, repetition_penalty=1.0, device="cuda"):
    """Yield the growing response text token-by-token (KV-cached)."""
    from .generate import sample_next_token

    ids = tok.encode(prompt) or [tok.eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    text = ""
    for _ in range(max_new_tokens):
        cond = idx if past is None else idx[:, -1:]
        logits, _, past = model(cond, past_kvs=past, use_cache=True)
        logits = logits[:, -1, :].float()
        if repetition_penalty != 1.0:
            for b in range(idx.shape[0]):
                seen = set(idx[b].tolist())
                for t in seen:
                    logits[b, t] = logits[b, t] / repetition_penalty if logits[b, t] > 0 \
                        else logits[b, t] * repetition_penalty
        nxt = sample_next_token(logits, temperature, top_k, top_p)
        if nxt.item() == tok.eos_id:
            break
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        text += tok.decode([nxt.item()])
        yield text
