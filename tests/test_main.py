import os
import subprocess

import pytest


def run_script(*args, **kwargs) -> subprocess.CompletedProcess:
    # Construct the command line arguments list
    cmd = ["tcr"]

    # Add flags without values
    if args:
        cmd.extend([f"--{arg}" for arg in args])

    # Add arguments with values
    for key, value in kwargs.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    # Run the script as a subprocess
    process = subprocess.run(cmd, text=True, capture_output=True)

    # Return the exit code
    return process


base_args = ["integration_test"]

base_kwargs = {
    "max_epochs": 1,
    "check_val_every_n_epoch": 1,
    "n_sim_batches": 16,
    "batch_size_sim": 8,
    "batch_size": 4,
    "weight_decay": 0.001,
    "balance_reg": 0.001,
    "overlap_reg": 0.001,
}


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ([*base_args], {"s": "linear", "n_vars_high_level": 1, **base_kwargs}),
        ([*base_args], {"s": "linear", "n_vars_high_level": 4, **base_kwargs}),
        ([*base_args], {"s": "simple_sparse", "n_vars_high_level": 1, **base_kwargs}),
        ([*base_args], {"s": "two_branch", "n_vars_high_level": 1, **base_kwargs}),
        ([*base_args], {"s": "double_well", "n_vars_high_level": 1, **base_kwargs}),
        ([*base_args], {"s": "triple_well", "n_vars_high_level": 1, **base_kwargs}),
    ],
)
def test_main(args, kwargs):
    os.environ["WANDB_MODE"] = "disabled"
    p = run_script(*args, **kwargs)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
    assert p.returncode == 0, "Script failed to run"
