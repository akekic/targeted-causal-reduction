import os

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from .config import CLASS, CLASS_ARGS, SIM_ARGS, LOW_LEVEL_MODELS
from .parser import ArgumentParser
from targeted_causal_reduction import TargetedCausalReduction
from targeted_causal_reduction.causal_model import HighLevelCausalModel
from targeted_causal_reduction.data_generators import (
    LinearCausalModel,
    ground_truth_solution,
    train_val_test_split,
    HDF5Dataset,
)
from targeted_causal_reduction.data_generators import (
    make_dataset,
    make_dataloaders,
)
from targeted_causal_reduction.reduction import Reduction


def main():
    global low_level, high_level
    args = ArgumentParser().parse_args()
    L.seed_everything(args.seed, workers=True)

    # 0. Initialize logger and checkpoint callback
    wandb_logger = WandbLogger(project="targeted-causal-reduction")
    wandb_logger.experiment.config.update(args.as_dict(), allow_val_change=True)
    print(f"Saving checkpoints to {wandb_logger.experiment.dir}")
    checkpoint_callback = ModelCheckpoint(
        dirpath=wandb_logger.experiment.dir,
        save_last=True,
        every_n_epochs=args.check_val_every_n_epoch,
    )

    # 1. Generate data
    low_level = LOW_LEVEL_MODELS[args.s][CLASS](**LOW_LEVEL_MODELS[args.s][CLASS_ARGS])
    if args.load_data:
        print(f"Loading data from {args.data_dir.absolute()}")
        hdf5_file = args.data_dir / "data.h5"
        if args.copy_to_tmp:
            print(f"Copying data to {args.tmp_dir.absolute()}")
            tmp_dir = args.tmp_dir
            tmp_dir.mkdir(exist_ok=True)
            os.system(f"cp {args.data_dir / 'data.h5'} {tmp_dir / 'data.h5'}")
            hdf5_file = tmp_dir / "data.h5"
        if args.on_disk:
            try:
                low_level_attrs = low_level.attributes()
            except (AttributeError, NotImplementedError):
                low_level_attrs = {}
            attrs = {**low_level_attrs, **LOW_LEVEL_MODELS[args.s][SIM_ARGS]}
            ds = HDF5Dataset(hdf5_file, attrs=attrs)
        else:
            ds = torch.load(args.data_dir / "dataset.pt")
        if args.save_data:
            print("Skipping saving data because load_data is True")
    else:
        ds = make_dataset(
            low_level,
            n_sim_batches=args.n_sim_batches,
            batch_size_sim=args.batch_size_sim,
            sim_args=LOW_LEVEL_MODELS[args.s][SIM_ARGS],
            dir=args.data_dir,
            use_multiprocessing=args.use_multiprocessing,
            in_memory=not args.on_disk,
            tmp_dir=args.tmp_dir,
        )
        if args.save_data and not args.on_disk:
            args.data_dir.mkdir(exist_ok=True)
            print(f"Saving data to {args.data_dir}")
            torch.save(ds, args.data_dir / "dataset.pt")
    datasets = train_val_test_split(ds)
    L.seed_everything(args.training_seed, workers=True)
    dataloaders = make_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
    )
    tau_map_ground_truth, omega_map_ground_truth = (
        ground_truth_solution(low_level)
        if type(low_level) == LinearCausalModel
        else (None, None)
    )

    # 2. Initialize reduction and tcr model
    high_level = HighLevelCausalModel(n_vars=args.n_vars_high_level)
    reduction = Reduction(
        low_level, high_level, tau_map_ground_truth, omega_map_ground_truth
    )
    tcr = TargetedCausalReduction(
        reduction,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_scheduler=args.lr_scheduler if args.lr_scheduler != "none" else None,
        weight_decay=args.weight_decay,
        overlap_reg=args.overlap_reg,
        balance_reg=args.balance_reg,
    )

    # 3. Train tcr model
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=[wandb_logger],
        fast_dev_run=args.integration_test,
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    trainer.fit(
        tcr, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"]
    )

    # 4. Evaluate tcr model
    if not args.integration_test:
        trainer.test(dataloaders=dataloaders["test"])


if __name__ == "__main__":
    main()
