from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .disk import _make_disk_dataset
from .in_memory import _make_in_memory_dataset
from ...causal_model import LowLevelCausalModel


def make_dataset(
    low_level: LowLevelCausalModel,
    n_sim_batches: int,
    batch_size_sim: int,
    sim_args: dict,
    use_multiprocessing: bool = True,
    in_memory: bool = True,
    dir: Optional[Path] = None,
    tmp_dir: Optional[Path] = None,
) -> Dataset:
    """
    Make torch.TensorDataset data set by sampling from the low-level causal model.

    Parameters
    ----------
    low_level: LowLevelCausalModel
        Low-level causal model.
    n_sim_batches: int
        Number of simulation batches.
    batch_size_sim: int
        Batch size for simulation. I.e. how many samples are simulated for a given intervention.
    sim_args: dict
        Keyword arguments for the low-level passed to low_level.sample().
    use_multiprocessing: bool
        Whether to use multiprocessing to speed up the sampling. Each batch is sampled in a separate process.
    in_memory: bool
        Whether to keep the data in memory or save it to disk.
    dir: Optional[Path]
        Directory where the data is saved to. Only used when in_memory=False.
    tmp_dir: Optional[Path]
        Directory where temporary files are saved to. Only used when in_memory=False.

    Returns
    -------
    dataset: Dataset
        Dataset containing the simulated data.
    """
    if in_memory:
        dataset = _make_in_memory_dataset(
            low_level,
            n_sim_batches,
            batch_size_sim,
            sim_args,
            use_multiprocessing,
        )
    else:
        assert dir is not None, "dir must be specified when in_memory=False"
        dataset = _make_disk_dataset(
            low_level,
            n_sim_batches,
            batch_size_sim,
            sim_args,
            dir=dir,
            tmp_dir=tmp_dir,
            use_multiprocessing=use_multiprocessing,
        )
    return dataset


def train_val_test_split(dataset: Dataset) -> dict[str, Dataset]:
    # split in training, validation and test set
    train_size = int(0.8 * len(dataset))
    val_size = int(0.5 * (len(dataset) - train_size))
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(
        dataset, range(train_size, (train_size + val_size))
    )
    test_dataset = torch.utils.data.Subset(
        dataset, range((train_size + val_size), len(dataset))
    )
    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def make_dataloaders(
    datasets: dict[str, Dataset],
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    dataloaders = {}
    for key, dataset in datasets.items():
        shuffle = key == "train"
        dataloaders[key] = DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    return dataloaders
