from pathlib import Path
from typing import Literal

from tap import Tap


class ArgumentParser(Tap):
    s: Literal[
        "double_well",
        "triple_well",
        "simple_sparse",
        "linear",
        "chain",
        "chain_masked",
        "star",
        "two_branch",
        "mass_spring",
        "mass_spring_grouped",
        "mass_no_spring",
    ] = "linear"
    """The simulator to use."""

    max_epochs: int = 2
    """The maximum number of epochs to train the model."""

    batch_size: int = 8
    """
    The batch size in the dataloader, i.e. number of distinct intervention settings passed to the model per batch.
    The total number of samples in a batch is batch_size * batch_size_sim.
    """

    batch_size_sim: int = 16
    """The number of simulations to run per intervention setting."""
    # TODO: add this as a constraint, see https://github.com/swansonk14/typed-argument-parser#argument-processing

    n_sim_batches: int = 128
    """
    The number of batches of simulations to run. One simulation batch contains batch_size_sim simulations.
    This also corresponds to the number of different intervention settings.
    """

    lr: float = 1e-3
    """The learning rate."""

    lr_min: float = 1e-5
    """The minimum learning rate for the cosine learning rate scheduler."""

    lr_scheduler: Literal["none", "cosine"] = "none"
    """The learning rate scheduler to use."""

    weight_decay: float = 1e-2
    """The weight decay applied to tau and omega maps."""

    overlap_reg: float = 0.0
    """The weight of the overlap regularization."""

    balance_reg: float = 0.0
    """The weight of the balance regularization."""

    n_vars_high_level: int = 1
    """The number of causes in the high-level causal model."""

    check_val_every_n_epoch: int = 1
    """How often to check the validation set during training."""

    save_data: bool = False
    """Whether to save the datasets."""

    data_dir: Path = Path("../data/")
    """The directory to save the dataloaders to."""

    tmp_dir: Path = Path("../data/tmp")
    """The directory to save the temporary files to. Only used if in_memory is False."""

    load_data: bool = False
    """Whether to load the datasets from disk."""

    use_multiprocessing: bool = False
    """Whether to use multiprocessing to generate the simulation data."""

    on_disk: bool = False
    """Whether to save the simulation data on disk. Then the data is not loaded into memory."""

    copy_to_tmp: bool = False
    """Whether to copy the data to the tmp directory before training."""

    seed: int = 42
    """The random seed to use."""

    training_seed: int = 42
    """The random seed to use for training."""

    integration_test: bool = False
    """Run in test mode."""
