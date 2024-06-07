import concurrent.futures
import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from tqdm import tqdm

from .hdf5_dataset import HDF5Dataset
from .in_memory import _simulate_batch
from ...causal_model import LowLevelCausalModel


def _simulate_batch_disk(
    low_level, batch_size_sim, sim_args, tmp_dir: Path, idx: int, seed: int
):
    np.random.seed(seed)
    intervention = (
        low_level.zero_intervention() if idx == 0 else low_level.sample_intervention()
    )
    X, I, Y = _simulate_batch(intervention, low_level, batch_size_sim, sim_args)
    with h5py.File(tmp_dir / f"{idx}.h5", "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("I", data=I)
        f.create_dataset("Y", data=Y)
    return


def _cleanup_simulation_files(tmp_dir, n_sim_batches):
    with tqdm(
        total=n_sim_batches,
        desc="Cleaning up simulation files",
        miniters=int(n_sim_batches / 100),
        unit="sim batch",
    ) as pbar:
        for idx in range(n_sim_batches):
            os.remove(tmp_dir / f"{idx}.h5")
            pbar.update(1)


def _merge_datasets_disk(tmp_dir: Path, dir: Path, n_sim_batches: int) -> None:
    with h5py.File(dir / "data.h5", "w") as merged_dataset:
        first_idx = 0
        with h5py.File(tmp_dir / f"{first_idx}.h5", "r") as dataset:
            shape_X = (n_sim_batches,) + dataset.get("X").shape[1:]
            shape_I = (n_sim_batches,) + dataset.get("I").shape[1:]
            shape_Y = (n_sim_batches,) + dataset.get("Y").shape[1:]

            merged_dataset.create_dataset(
                "X",
                shape=shape_X,
                dtype=dataset.get("X").dtype,
            )
            merged_dataset.create_dataset(
                "I",
                shape=shape_I,
                dtype=dataset.get("I").dtype,
            )
            merged_dataset.create_dataset(
                "Y",
                shape=shape_Y,
                dtype=dataset.get("Y").dtype,
            )

        with tqdm(
            total=n_sim_batches,
            desc="Merging simulation files",
            miniters=int(n_sim_batches / 100),
            unit="sim batch",
        ) as pbar:
            for i in range(n_sim_batches):
                with h5py.File(tmp_dir / f"{i}.h5", "r") as dataset:
                    merged_dataset["X"][i : i + 1] = dataset["X"]
                    merged_dataset["I"][i : i + 1] = dataset["I"]
                    merged_dataset["Y"][i : i + 1] = dataset["Y"]
                pbar.update(1)


def _write_metadata(dir: Path, low_level: LowLevelCausalModel, sim_args: dict) -> None:
    """
    Store the attributes defined in low_level and sim_args as metadata in the HDF5 file.
    """
    try:
        low_level_attrs = low_level.attributes()
    except (AttributeError, NotImplementedError):
        low_level_attrs = {}
    attrs = {**low_level_attrs, **sim_args}
    with h5py.File(dir / "data.h5", "a") as f:
        for key, value in attrs.items():
            f.attrs[key] = value


def _make_disk_dataset(
    low_level: LowLevelCausalModel,
    n_sim_batches: int,
    batch_size_sim: int,
    sim_args: dict,
    dir: Path,
    tmp_dir: Optional[Path] = None,
    copy_to_tmp: bool = False,
    use_multiprocessing: bool = True,
) -> HDF5Dataset:
    """
    Generate a dataset and save it to disk.

    Each process generates a batch of data and saves a separate temporary file to disk. The temporary files are then
    merged into a single file and stored in dir.

    Parameters
    ----------
    low_level: LowLevelCausalModel
        Low-level causal model.
    n_sim_batches: int
        Number of simulation batches. Each batch is simulated with a different intervention.
    batch_size_sim: int
        Batch size for simulation. I.e. how many samples are simulated for a given intervention.
    sim_args: dict
        Keyword arguments for the low-level passed to low_level.sample().
    dir: Path
        Directory where the data is saved to.
    tmp_dir: Optional[Path]
        Directory where temporary files are saved to.
    copy_to_tmp: bool
        Whether to copy the data to the tmp directory before training. If true, the final merged data is saved to dir
        and then copied to tmp_dir and loaded from there during training. This is useful for speeding up training when
        the data is stored on a network drive. This also ensures that two separate training runs do not interfere with
        each other by reading the same .h5 file.
    use_multiprocessing: bool
        Whether to use multiprocessing to speed up the sampling. Each batch is sampled in a separate process.

    Returns
    -------
    dataset: HDF5Dataset
        Dataset containing the simulated data.
    """
    if tmp_dir is None:
        tmp_dir = dir / "tmp"
        tmp_dir.mkdir(exist_ok=True)

    dir.mkdir(exist_ok=True)
    with tqdm(
        total=n_sim_batches,
        desc="Generating simulation data",
        miniters=int(n_sim_batches / 100),
        unit="sim batch",
    ) as pbar:
        if use_multiprocessing:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                seed_min, seed_max = 0, 2**32 - 1
                random_seeds = np.random.randint(seed_min, seed_max, size=n_sim_batches)
                # noinspection PyTypeChecker
                futures = {
                    executor.submit(
                        _simulate_batch_disk,
                        low_level,
                        batch_size_sim,
                        sim_args,
                        tmp_dir,
                        idx,
                        random_seeds[idx],
                    ): idx
                    for idx in range(n_sim_batches)
                }
                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)
                    try:
                        future.result()
                    except Exception as e:
                        idx = futures[future]
                        print(f"Failed to generate batch {idx}: {e}")
        else:
            for idx in range(n_sim_batches):
                _simulate_batch_disk(
                    low_level,
                    batch_size_sim,
                    sim_args,
                    tmp_dir,
                    idx,
                    seed=idx,
                )
                pbar.update(1)

    _merge_datasets_disk(tmp_dir, dir, n_sim_batches)
    _write_metadata(dir, low_level, sim_args)
    _cleanup_simulation_files(tmp_dir, n_sim_batches)
    hdf5_file = dir / "data.h5"
    if copy_to_tmp:
        tmp_dir.mkdir(exist_ok=True)
        os.system(f"cp {dir / 'data.h5'} {tmp_dir / 'data.h5'}")
        hdf5_file = tmp_dir / "data.h5"
    dataset = HDF5Dataset(path=hdf5_file)
    return dataset
