"""Resumable downloader with optional HuggingFace mirror (hf-mirror.com)."""
import argparse
import os
import sys

import requests


def download(url, out, mirror=False, chunk=8 * 1024 * 1024):
    if mirror and "huggingface.co" in url:
        url = url.replace("huggingface.co", "hf-mirror.com")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pos = os.path.getsize(out) if os.path.exists(out) else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        if pos and r.status_code == 200:  # server ignored Range
            pos = 0
        total = int(r.headers.get("Content-Length", 0)) + pos
        mode = "ab" if pos else "wb"
        done = pos
        with open(out, mode) as f:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    sys.stdout.write(f"\r{done/1e9:.2f} GB / {total/1e9:.2f} GB ({pct:.1f}%)")
                    sys.stdout.flush()
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mirror", action="store_true")
    args = p.parse_args()
    download(args.url, args.out, args.mirror)
