"""PyTorch dataset and dataloader utilities for windowed C-MAPSS data."""

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEED = 42


class CMAPSSDataset(Dataset):
    """A :class:`~torch.utils.data.Dataset` over windowed C-MAPSS tensors.

    Wraps a feature tensor of shape ``(n_windows, window_size, n_features)``
    and a target tensor of shape ``(n_windows,)``.
    """

    def __init__(
        self, X: torch.Tensor, y: torch.Tensor, mode: str = "train"
    ) -> None:
        """Initialize the dataset.

        Args:
            X: Feature tensor of shape ``(n_windows, window_size, n_features)``.
            y: Target tensor of shape ``(n_windows,)``.
            mode: Dataset role, one of ``train``/``val``/``test``.

        Raises:
            ValueError: If ``X`` and ``y`` have mismatched lengths.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y length mismatch: {X.shape[0]} != {y.shape[0]}"
            )
        self.X = X.float()
        self.y = y.float()
        self.mode = mode
        logger.info(
            "Initialized %s dataset with %d samples", mode, self.X.shape[0]
        )

    def __len__(self) -> int:
        """Return the number of windows in the dataset.

        Returns:
            The sample count.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the feature window and target at a given index.

        Args:
            idx: Sample index.

        Returns:
            A tuple ``(features, target)``.
        """
        return self.X[idx], self.y[idx]


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate a list of samples into stacked batch tensors.

    Args:
        batch: List of ``(features, target)`` tuples.

    Returns:
        A tuple ``(X, y)`` with batched feature and target tensors.
    """
    features = torch.stack([item[0] for item in batch], dim=0)
    targets = torch.stack([item[1].reshape(()) for item in batch], dim=0)
    return features, targets


def get_dataloader(
    dataset: CMAPSSDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a :class:`~torch.utils.data.DataLoader` over a dataset.

    Args:
        dataset: The dataset to iterate.
        batch_size: Number of windows per batch.
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of worker processes for loading.

    Returns:
        A configured ``DataLoader`` with a seeded shuffle generator.
    """
    generator = torch.Generator()
    generator.manual_seed(_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        generator=generator,
        drop_last=False,
    )
