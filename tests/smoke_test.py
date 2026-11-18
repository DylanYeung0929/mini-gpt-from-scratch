"""CPU-friendly smoke test of model, KV-cache, LoRA and data pipeline."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from gptfs.config import GPTConfig
from gptfs.model import GPT
from gptfs.lora import LoRALinear, apply_lora, merge_lora


def test_forward_backward():
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4,
                    n_kv_head=2, n_embd=64, ffn_mult=4)
    model = GPT(cfg)
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    logits, loss, _ = model(x, y)
    assert logits.shape == (2, 32, 256)
    loss.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"forward/backward OK | loss={loss.item():.4f} | params={model.num_params()} | grad_tensors={grads}")


def test_kv_cache():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=128, block_size=64, n_layer=2, n_head=4,
                    n_kv_head=4, n_embd=64, ffn_mult=4)
    model = GPT(cfg).eval()
    x = torch.randint(0, 128, (1, 16))
    full, _, _ = model(x)
    # incremental decode
    outs = []
    past = None
    for t in range(16):
        cur = x[:, :1] if t == 0 else x[:, t:t + 1]
        lg, _, past = model(cur, past_kvs=past, use_cache=True)
        outs.append(lg[:, -1])
    inc = torch.stack(outs, dim=1)
    err = (full - inc).abs().max().item()
    assert err < 1e-4, err
    print(f"KV-cache consistency OK | max diff={err:.2e}")


def test_lora():
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4,
                    n_kv_head=4, n_embd=32, ffn_mult=2)
    model = GPT(cfg)
    model.eval()
    x = torch.randint(0, 64, (1, 16))
    before, _, _ = model(x)
    n = apply_lora(model, r=4, alpha=8)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(p, std=0.01)
    after, _, _ = model(x)
    assert not torch.allclose(before, after)
    merged = merge_lora(model)
    m, _, _ = merged(x)
    err = (m - after).abs().max().item()
    assert err < 1e-5, err
    print(f"LoRA apply/merge OK | trainable={n} | merge diff={err:.2e}")


def test_tokenizer_and_data():
    from gptfs.tokenizer import train_bpe, TokenizerWrapper
    from gptfs.data import prepare_token_bin, get_batch
    from tokenizers import Tokenizer

    with tempfile.TemporaryDirectory() as d:
        text_path = os.path.join(d, "corpus.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            for i in range(2000):
                f.write(f"Story {i}. Once upon a time there was a little cat. "
                        f"It liked to play with a red ball. The end.<|endoftext|>")
        tok_dir = os.path.join(d, "tok")
        train_bpe([text_path], tok_dir, vocab_size=512, sample_chars=100000)
        tok = TokenizerWrapper(tok_dir)
        meta = prepare_token_bin(text_path, os.path.join(d, "out.bin"), tok, verbose=False)
        assert meta["tokens"] > 0
        meta2 = prepare_token_bin(text_path, os.path.join(d, "val.bin"), tok, verbose=False)
        text = "Hello world 你好"
        assert tok.decode(tok.encode(text)) == text, "tokenizer round-trip is not lossless"
        data = np.memmap(os.path.join(d, "out.bin"), dtype=np.uint16, mode="r")
        x, y = get_batch(data, block_size=16, batch_size=2, device=torch.device("cpu"))
        assert tuple(x.shape) == (2, 16) and tuple(y.shape) == (2, 16)
        assert int(x.min()) >= 0 and int(x.max()) < tok.vocab_size
        data._mmap.close()          # release the file handle before tempdir cleanup (Windows)
        print(f"tokenizer+data OK | vocab={tok.vocab_size} | tokens={meta['tokens']}"
              f" | roundtrip OK | window={tuple(x.shape)}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_forward_backward()
    test_kv_cache()
    test_lora()
    test_tokenizer_and_data()
    print("ALL SMOKE TESTS PASSED")
