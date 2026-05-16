"""
data_loader.py — Handles downloading Tiny Shakespeare (or any HF text dataset),
building the character vocabulary, and turning text into PyTorch DataLoaders.

I cache everything because re-downloading + re-encoding every time was annoying.
"""

import os
import pickle
import logging
from typing import Tuple, List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split

logger = logging.getLogger(__name__)


class CharVocab:
    """Maps characters <-> indices. Simple but does the job."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char2idx: Dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.idx2char: Dict[int, str] = {i: c for c, i in self.char2idx.items()}
        self.size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices) -> str:
        return "".join(self.idx2char.get(int(i), "?") for i in indices)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CharVocab":
        with open(path, "rb") as f:
            return pickle.load(f)


class CharDataset(Dataset):
    """
    Sliding window dataset.
    For every position i, input is chars[i:i+seq], target is chars[i+1:i+seq+1].
    This is the classic setup for next-char prediction.
    """

    def __init__(self, data: torch.Tensor, seq_length: int):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_length]
        y = self.data[idx + 1 : idx + self.seq_length + 1]
        return x, y


def load_data(
    dataset_name: str = "Trelis/tiny-shakespeare",
    split: str = "train",
    text_column: str = "text",
    max_chars: Optional[int] = None,
    seq_length: int = 100,
    batch_size: int = 64,
    valid_split: float = 0.1,
    cache_dir: str = "data_cache",
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, CharVocab]:
    """
    Main entry point. Downloads from HF if needed, builds vocab, caches to disk,
    then gives you train/val loaders + the vocab object.
    """

    vocab_path = os.path.join(cache_dir, "vocab.pkl")
    data_path = os.path.join(cache_dir, "data.pt")

    if os.path.exists(vocab_path) and os.path.exists(data_path):
        logger.info("Loading preprocessed data from cache …")
        vocab = CharVocab.load(vocab_path)
        data = torch.load(data_path)
    else:
        logger.info("Downloading dataset '%s' from HuggingFace …", dataset_name)
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "Install the 'datasets' library: pip install datasets"
            ) from e

        os.makedirs(cache_dir, exist_ok=True)
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)

        # Join everything into one big string
        if text_column in ds.column_names:
            raw_text = "\n".join(str(row) for row in ds[text_column])
        else:
            raw_text = "\n".join(
                " ".join(str(v) for v in row.values()) for row in ds
            )

        if max_chars:
            raw_text = raw_text[:max_chars]

        logger.info("Corpus size: %s characters", f"{len(raw_text):,}")

        vocab = CharVocab(raw_text)
        logger.info("Vocabulary size: %d unique characters", vocab.size)

        encoded = vocab.encode(raw_text)
        data = torch.tensor(encoded, dtype=torch.long)

        vocab.save(vocab_path)
        torch.save(data, data_path)
        logger.info("Cache saved to '%s'", cache_dir)

    # Create sliding-window datasets
    full_ds = CharDataset(data, seq_length)
    n_valid = max(1, int(len(full_ds) * valid_split))
    n_train = len(full_ds) - n_valid

    train_ds, valid_ds = random_split(
        full_ds,
        [n_train, n_valid],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    logger.info(
        "Train batches: %d | Val batches: %d",
        len(train_loader),
        len(valid_loader),
    )
    return train_loader, valid_loader, vocab
