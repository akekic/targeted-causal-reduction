from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """
    Dataset that loads data from an HDF5 file. The file must contain the following datasets:
        - X: torch.Tensor of shape (n_sim_batches, batch_size_sim, n_vars)
        - I: torch.Tensor of shape (n_sim_batches, batch_size_sim, n_vars)
        - Y: torch.Tensor of shape (n_sim_batches, batch_size_sim, 1)

    It loads the data directly from disk and does not keep it in memory.
    """

    def __init__(self, path: Path, attrs: Optional[dict] = None) -> None:
        """
        Parameters
        ----------
        path: Path
            Path to the HDF5 file.
        attrs: Optional[dict]
            Dictionary of attributes of the dataset. This is checked against the attributes in the HDF5 file.
        """
        self.path = path
        with h5py.File(path, "r") as f:
            self.length = f["X"].shape[0]
            if attrs is not None:
                for key, value in attrs.items():
                    assert key in f.attrs, f"Attribute {key} not found in {path}"
                    # if f.attrs[key] is an array, check if all elements are close to value

                    if isinstance(f.attrs[key], (list, tuple, np.ndarray)):
                        assert np.allclose(
                            f.attrs[key], value
                        ), f"Attribute {key} in {path} has value {f.attrs[key]} but should be {value}"
                    else:
                        assert (
                            f.attrs[key] == value
                        ), f"Attribute {key} in {path} has value {f.attrs[key]} but should be {value}"

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.path, "r") as f:
            return (
                torch.from_numpy(f["X"][idx]),
                torch.from_numpy(f["I"][idx]),
                torch.from_numpy(f["Y"][idx]),
            )
