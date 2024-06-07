import concurrent.futures
from functools import partial

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm

from ...causal_model import LowLevelCausalModel


def _simulate_batch(
    intervention: np.ndarray,
    low_level: LowLevelCausalModel,
    batch_size_sim: int,
    sim_args: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    X, I, Y = low_level.sample(intervention, batch_size_sim, **sim_args)
    X = torch.from_numpy(X).to(torch.float32)
    I = torch.from_numpy(I).to(torch.float32)
    Y = torch.from_numpy(Y).to(torch.float32)
    return X, I, Y


def _make_in_memory_dataset(
    low_level: LowLevelCausalModel,
    n_sim_batches: int,
    batch_size_sim: int,
    sim_args: dict,
    use_multiprocessing: bool = True,
) -> Dataset:
    interventions = [low_level.zero_intervention()] + [
        low_level.sample_intervention() for _ in range(n_sim_batches - 1)
    ]
    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            partial_simulate_batch = partial(
                _simulate_batch,
                low_level=low_level,
                batch_size_sim=batch_size_sim,
                sim_args=sim_args,
            )
            data_X, data_I, data_Y = zip(
                *tqdm(
                    executor.map(
                        partial_simulate_batch,
                        interventions,
                    ),
                    desc="Generating simulation data",
                    miniters=int(n_sim_batches / 100),
                    unit="sim batch",
                    total=n_sim_batches,
                )
            )
    else:
        data_X, data_I, data_Y = [], [], []
        for intervention in tqdm(
            interventions,
            desc="Generating simulation data",
            miniters=int(n_sim_batches / 100),
            unit="sim batch",
            total=n_sim_batches,
        ):
            X, I, Y = _simulate_batch(
                intervention,
                low_level=low_level,
                batch_size_sim=batch_size_sim,
                sim_args=sim_args,
            )
            data_X.append(X)
            data_I.append(I)
            data_Y.append(Y)
    data_X = torch.cat(data_X)
    data_I = torch.cat(data_I)
    data_Y = torch.cat(data_Y)
    dataset = TensorDataset(data_X, data_I, data_Y)
    return dataset
