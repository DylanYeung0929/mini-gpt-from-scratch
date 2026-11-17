"""Corpus preparation (text -> uint16 token stream) and training datasets."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def prepare_token_bin(text_path, out_path, tokenizer, doc_separator="<|endoftext|>",
                      eos_id=None, batch_docs=4096, add_eos=True, verbose=True):
    """Tokenize a text file (documents separated by `doc_separator`) into a uint16 .bin."""
    eos_id = tokenizer.eos_id if eos_id is None else eos_id
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_docs = total_tokens = 0
    buf = []

    def flush(buf):
        nonlocal total_docs, total_tokens
        encs = tokenizer.encode_batch(buf)
        ids = []
        for enc, doc in zip(encs, buf):
            ids.extend(enc.ids)
            if add_eos:
                ids.append(eos_id)
            total_docs += 1
        arr = np.array(ids, dtype=np.uint16)
        with open(out_path, "ab") as f:
            f.write(arr.tobytes())
        total_tokens += len(arr)
        if verbose and total_docs % (batch_docs * 10) < batch_docs:
            print(f"  {total_docs:,} docs | {total_tokens/1e6:.1f}M tokens", flush=True)

    # truncate existing file first
    open(out_path, "wb").close()
    pending = ""
    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            chunk = f.read(8_000_000)
            if not chunk:
                break
            pending += chunk
            parts = pending.split(doc_separator)
            pending = parts.pop()          # last part may be incomplete
            buf.extend(parts)
            while len(buf) >= batch_docs:
                flush(buf[:batch_docs])
                del buf[:batch_docs]
    # flush remainder
    if pending.strip():
        buf.append(pending)
    for i in range(0, len(buf), batch_docs):
        flush(buf[i:i + batch_docs])

    meta = {"tokens": total_tokens, "docs": total_docs, "dtype": "uint16"}
    import json
    with open(out_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


class TokenDataset(Dataset):
    """Random-offset next-token chunks over a memmapped uint16 token stream."""

    def __init__(self, bin_path, block_size):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.block_size = block_size
        self.n_tokens = len(self.data)

    def __len__(self):
        return max(1, self.n_tokens // self.block_size)

    def __getitem__(self, _):
        i = np.random.randint(0, self.n_tokens - self.block_size - 1)
        chunk = np.asarray(self.data[i: i + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i: i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1: i + 1 + block_size] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
