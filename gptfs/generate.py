"""Sampling utilities and text generation entry points."""
import torch


def sample_next_token(logits, temperature=0.8, top_k=None, top_p=None):
    """logits: (B, V) -> (B,) token ids."""
    if temperature <= 0:  # greedy
        return logits.argmax(dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = torch.where(logits < thresh, float("-inf"), logits)
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        mask = cum - probs > top_p          # keep first token above threshold
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8,
                  top_k=50, top_p=0.95, repetition_penalty=1.0, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.eos_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
        top_p=top_p, repetition_penalty=repetition_penalty, eos_id=tokenizer.eos_id,
    )
    new_ids = out[0].tolist()[len(ids):]
    # strip special tokens from display
    text = tokenizer.decode(new_ids)
    for sp in ["<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]:
        text = text.replace(sp, "")
    return text.strip()


def build_chat_prompt(user_msg, system=None):
    s = ""
    if system:
        s += f"<|system|>\n{system}\n"
    s += f"<|user|>\n{user_msg}\n<|assistant|>\n"
    return s
