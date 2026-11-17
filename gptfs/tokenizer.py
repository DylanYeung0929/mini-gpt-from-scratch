"""BPE tokenizer training / loading (built on HuggingFace `tokenizers`)."""
import json
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"]


def train_bpe(text_files, out_dir, vocab_size=32768, sample_chars=300_000_000):
    """Train a byte-level BPE tokenizer on (a sample of) the given text files."""
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        min_frequency=2,
        show_progress=True,
    )

    def text_iter():
        budget = sample_chars
        for path in text_files:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                while budget > 0:
                    chunk = f.read(1_000_000)
                    if not chunk:
                        break
                    budget -= len(chunk)
                    yield chunk

    tok.train_from_iterator(text_iter(), trainer=trainer)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    meta = {"vocab_size": tok.get_vocab_size(), "special_tokens": SPECIAL_TOKENS}
    with open(os.path.join(out_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return tok


class TokenizerWrapper:
    def __init__(self, tokenizer_dir):
        self.tok = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.eos_id = self.tok.token_to_id("<|endoftext|>")
        self.vocab_size = self.tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts):
        return self.tok.encode_batch(texts, add_special_tokens=False)

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=False)
