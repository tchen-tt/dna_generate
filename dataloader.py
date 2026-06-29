"""Data loading and preprocessing for DNA sequences.

Data format: tab-separated, columns chr / sequence / TAG
Chromosome split: chr1=test set, chr2=validation set, others=training set
Encoding: one-hot, 0 replaced with -1 for symmetric [-1, +1] signal distribution
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from constants import NUCLEOTIDES, SEQ_LEN
from utils import one_hot_encode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset(
    data_path: str,
    saved_data_path: str,
    load_saved_data: bool = True,
    debug: bool = False,
) -> tuple[Dataset, Dataset, list[int], dict[int, str]]:
    """Load and prepare train/validation datasets.

    Args:
        data_path: Path to raw TSV data file
        saved_data_path: Path to cached preprocessed data pickle
        load_saved_data: Whether to load from cache if available
        debug: If True, use only one sample for fast iteration

    Returns:
        Tuple of (train_dataset, val_dataset, cell_type_ids, id_to_name_mapping)
    """
    encode_data = _load_encode_data(data_path, saved_data_path, load_saved_data)

    if debug:
        x_train = encode_data["X_train"][:1]
        y_train = encode_data["x_train_cell_type"][:1]
        x_val   = encode_data["X_val"][:1]
        y_val   = encode_data["x_val_cell_type"][:1]
    else:
        x_train = encode_data["X_train"]
        y_train = encode_data["x_train_cell_type"]
        x_val   = encode_data["X_val"]
        y_val   = encode_data["x_val_cell_type"]

    train_dataset = SequenceDataset(x_train, y_train)
    val_dataset   = SequenceDataset(x_val,   y_val)

    return train_dataset, val_dataset, encode_data["cell_types"], encode_data["numeric_to_tag"]


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 2,
    distributed: bool = False,
    pin_memory: bool = True,
) -> tuple[DataLoader, Any]:
    """Create DataLoader with optional distributed sampler.

    Args:
        dataset: PyTorch Dataset instance
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for data loading
        distributed: Whether to use DistributedSampler for multi-GPU training
        pin_memory: Whether to pin memory for faster GPU transfer

    Returns:
        Tuple of (dataloader, sampler) where sampler is None for non-distributed
    """
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_encode_data(data_path: str, saved_data_path: str, load_saved_data: bool) -> dict:
    """Load preprocessed pickle or run preprocessing from scratch."""
    if load_saved_data and Path(saved_data_path).exists():
        with open(saved_data_path, "rb") as f:
            split_data = pickle.load(f)
    else:
        split_data = _preprocess_and_save(data_path, saved_data_path)

    return _encode_splits(split_data)


def _preprocess_and_save(input_path: str, output_path: str | None) -> dict:
    """Split raw TSV by chromosome and optionally cache to disk."""
    df = pd.read_csv(input_path, sep="\t")
    split = {
        "train_df":      df[(df["chr"] != "chr1") & (df["chr"] != "chr2")].reset_index(drop=True),
        "validation_df": df[df["chr"] == "chr2"].reset_index(drop=True),
        "test_df":       df[df["chr"] == "chr1"].reset_index(drop=True),
    }
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(split, f)
    return split


def _encode_splits(split_data: dict) -> dict:
    """One-hot encode sequences and build label mappings."""
    train_df = split_data["train_df"]
    val_df   = split_data["validation_df"]

    # Build label maps from training set (stable ordering)
    tag_to_numeric  = {tag: n for n, tag in enumerate(train_df["TAG"].unique(), 1)}
    numeric_to_tag  = {n: tag for tag, n in tag_to_numeric.items()}
    cell_types      = list(numeric_to_tag.keys())

    def encode_df(df: pd.DataFrame) -> np.ndarray:
        seqs = [
            one_hot_encode(seq, NUCLEOTIDES, SEQ_LEN)
            for seq in df["sequence"] if "N" not in seq
        ]
        arr = np.array([s.T for s in seqs], dtype=np.float32)  # (N, 4, 200)
        arr[arr == 0] = -1   # 0 → -1 for symmetric [-1, +1] signal
        return arr

    X_train = encode_df(train_df)
    X_val   = encode_df(val_df)

    # Filter val labels to match rows that passed the "N" check
    train_mask = [("N" not in s) for s in train_df["sequence"]]
    val_mask   = [("N" not in s) for s in val_df["sequence"]]

    y_train = torch.tensor([tag_to_numeric[t] for t, keep in zip(train_df["TAG"], train_mask) if keep])
    y_val   = torch.tensor([tag_to_numeric[t] for t, keep in zip(val_df["TAG"],   val_mask)   if keep])

    return {
        "X_train":            X_train,
        "X_val":              X_val,
        "x_train_cell_type":  y_train,
        "x_val_cell_type":    y_val,
        "tag_to_numeric":     tag_to_numeric,
        "numeric_to_tag":     numeric_to_tag,
        "cell_types":         cell_types,
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SequenceDataset(Dataset):
    """Wraps pre-encoded numpy arrays.

    The original implementation used T.ToTensor(), which has undefined behavior for float arrays
    (uint8 arrays are normalized to [0,1]). Here we directly convert to torch.Tensor to avoid
    ToTensor's implicit behavior.
    """

    def __init__(self, seqs: np.ndarray, labels: torch.Tensor) -> None:
        # seqs: (N, 4, 200) float32 with values in {-1, +1}
        self.seqs   = torch.from_numpy(seqs).unsqueeze(1)  # (N, 1, 4, 200)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.seqs[idx], self.labels[idx]
