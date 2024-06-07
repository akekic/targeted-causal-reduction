import os
import uuid
import warnings
from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from lightning import LightningModule
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .utils import batch_cov, ValidationStepOutputs
from ..data_generators.linear import plot_weights_two_branch
from ..data_generators.mass_spring import plot_weights_mass_spring
from ..reduction import Reduction


class TargetedCausalReduction(LightningModule):
    """
    The TargetedCausalReduction class is a LightningModule that implements the training, validation and testing steps
    for learning targeted causal reductions.

    Attributes
    ----------
    reduction : Reduction
        The reduction object that contains the low-level and high-level causal models, as well as the tau and omega
        maps that relate the low-level and high-level variables.
    lr : float
        The learning rate for the optimizer.
    lr_scheduler : Optional[str]
        The learning rate scheduler to use. If None, no learning rate scheduler is used. Options are "cosine" or None.
    lr_min : float
        The minimum learning rate for the cosine learning rate scheduler.
    weight_decay : float
        The weight decay applied to the parameters of the tau and omega maps.
    overlap_reg : float
        The weight of the overlap regularization term.
    balance_reg : float
        The weight of the balance regularization term.
    """

    def __init__(
        self,
        reduction: Reduction,
        lr: float = 1e-3,
        lr_scheduler: Optional[str] = None,
        lr_min: float = 1e-5,
        weight_decay: float = 1e-2,
        overlap_reg: float = 0.0,
        balance_reg: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.lr = lr
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.overlap_reg = overlap_reg
        self.balance_reg = balance_reg

        if overlap_reg > 0 and reduction.high_level.n_vars == 1:
            warnings.warn(
                "Overlap regularization is not effective with only one high level variable and is set to 0."
            )
            self.overlap_reg = 0
        if balance_reg > 0 and reduction.high_level.n_vars == 1:
            warnings.warn(
                "Balance regularization is not effective with only one high level variable and is set to 0."
            )
            self.balance_reg = 0

        self.validation_step_outputs = ValidationStepOutputs([], [], [], [])

    def forward(self, x, s):
        """
        Maps the low-level variables x and the low-level shift interventions s to the abstract variables z, the
        abstract target variable y and the abstract shift interventions r; i.e. x, s -> z, y, r.
        """
        z, r, y_hat = self.reduction(x, s)
        return z, r, y_hat

    def training_step(self, batch, batch_id):
        loss, _ = self._training_step(batch)
        return loss

    def _training_step(self, batch, logging_prefix="train"):
        x, s, y = batch
        z, r, y_hat = self(x, s)
        n_vars_high_level = self.reduction.high_level.n_vars
        mu_z = self.reduction.high_level.mu_z
        mu_hat_z = z.mean(axis=1, keepdim=True)
        sigma_z = self.reduction.high_level.sigma_z
        sigma_ygivz = self.reduction.high_level.sigma_y_given_z
        f_z = y_hat
        mu_hat_y = y.mean(axis=1, keepdim=True)
        total_cov = batch_cov(torch.concatenate((y, z), dim=-1))
        sigma_hat_y = torch.sqrt(total_cov[..., 0, 0])
        Sigma_hat_z = total_cov[..., 1:, 1:]
        cov_hat_y_z = total_cov[..., 1:, [0]]
        lstsq_z = (mu_z + r[:, [0], :] - mu_hat_z) ** 2 / sigma_z**2
        c_z = (
            0.5
            * (
                (
                    lstsq_z
                    + torch.diagonal(Sigma_hat_z, dim1=-2, dim2=-1) / sigma_z**2
                ).sum(dim=-1)
                - torch.log(torch.det(Sigma_hat_z) / torch.prod(sigma_z**2))
                - n_vars_high_level
            ).mean()
        )
        lstsq_ygivz = (
            f_z
            - mu_hat_y
            - (
                cov_hat_y_z.transpose(-2, -1)
                @ torch.inverse(Sigma_hat_z)
                @ (z - mu_hat_z).unsqueeze(-1)
            ).squeeze(-1)
        ) ** 2 / sigma_ygivz**2
        a = (
            (
                sigma_hat_y**2
                - (
                    cov_hat_y_z.transpose(-2, -1)
                    @ torch.inverse(Sigma_hat_z)
                    @ cov_hat_y_z
                ).squeeze(-1, -2)
                + 1e-7
            )
            / sigma_ygivz**2
        ).unsqueeze(-1)
        c_ygivz = 0.5 * (lstsq_ygivz + a - torch.log(a) - 1).mean()
        consistency = c_z + c_ygivz

        if self.overlap_reg == 0:
            overlap = torch.zeros_like(consistency)
        else:
            # go over all pairs of high level variables and compute the overlap
            overlap = 0
            for i in range(n_vars_high_level):
                for j in range(i + 1, n_vars_high_level):
                    tauvec_i = self.reduction.tau_map.get_weights(idx=i)
                    tauvec_j = self.reduction.tau_map.get_weights(idx=j)
                    overlap += (
                        torch.abs(tauvec_i).dot(torch.abs(tauvec_j))
                        / torch.sqrt(torch.sum(tauvec_i**2))
                        / torch.sqrt(torch.sum(tauvec_j**2))
                    )
                    omegavec_i = self.reduction.omega_map.get_weights(idx=i)
                    omegavec_j = self.reduction.omega_map.get_weights(idx=j)
                    overlap += (
                        torch.abs(omegavec_i).dot(torch.abs(omegavec_j))
                        / torch.sqrt(torch.sum(omegavec_i**2))
                        / torch.sqrt(torch.sum(omegavec_j**2))
                    )
        if self.balance_reg == 0:
            balance = torch.zeros_like(consistency)
        else:
            balance_tau_numerator = 0
            balance_tau_denominator = 0
            balance_omega_numerator = 0
            balance_omega_denominator = 0
            # get high level mechanism coefficients
            coeffs = self.reduction.high_level.mlp[0].weight.squeeze()
            for i in range(n_vars_high_level):
                tauvec = coeffs[i] * self.reduction.tau_map.get_weights(idx=i)
                omegavec = coeffs[i] * self.reduction.omega_map.get_weights(idx=i)
                balance_tau_numerator += torch.sum(tauvec**2)
                balance_tau_denominator += torch.sqrt(torch.sum(tauvec**2))
                balance_omega_numerator += torch.sum(omegavec**2)
                balance_omega_denominator += torch.sqrt(torch.sum(omegavec**2))
            balance_tau = torch.sqrt(balance_tau_numerator) / balance_tau_denominator
            balance_omega = (
                torch.sqrt(balance_omega_numerator) / balance_omega_denominator
            )
            balance = (balance_tau + balance_omega) - 2.0 / sqrt(n_vars_high_level)
        loss = consistency + self.overlap_reg * overlap + self.balance_reg * balance
        results = {
            "consistency_loss": consistency,
            "overlap_loss": overlap,
            "balance_loss": balance,
            "loss": loss,
            "lstsq_z": lstsq_z.mean(),
            "lstsq_ygivz": lstsq_ygivz.mean(),
            "a": a.mean(),
            "c_z": c_z,
            "c_ygivz": c_ygivz,
        }
        self.log(f"{logging_prefix}_consistency_loss", consistency, prog_bar=False)
        self.log(f"{logging_prefix}_overlap_loss", overlap, prog_bar=False)
        self.log(f"{logging_prefix}_balance_loss", balance, prog_bar=False)
        self.log(f"{logging_prefix}_loss", loss, prog_bar=True)
        self.log(f"{logging_prefix}_lstsq_z", lstsq_z.mean(), prog_bar=False)
        self.log(f"{logging_prefix}_lstsq_ygivz", lstsq_ygivz.mean(), prog_bar=False)
        self.log(f"{logging_prefix}_a", a.mean(), prog_bar=False)
        self.log(f"{logging_prefix}_c_z", c_z, prog_bar=False)
        self.log(f"{logging_prefix}_c_ygivz", c_ygivz, prog_bar=False)
        return loss, results

    def validation_step(self, batch, batch_idx):
        # return all validation data
        x, s, y = batch
        z, r, y_hat = self(x, s)

        loss, _ = self._training_step(batch, logging_prefix="val")

        self.validation_step_outputs.z_flat.append(
            torch.flatten(z, start_dim=0, end_dim=1)
        )
        self.validation_step_outputs.r_flat.append(
            torch.flatten(r, start_dim=0, end_dim=1)
        )
        self.validation_step_outputs.y_flat.append(
            torch.flatten(y, start_dim=0, end_dim=1)
        )
        self.validation_step_outputs.loss.append(loss)

    def on_validation_epoch_end(self) -> None:
        # compute mean validation loss
        mean_val_loss = torch.stack(self.validation_step_outputs.loss).mean()
        self.log("val_mean_loss", mean_val_loss, prog_bar=False)

        # compute L2 loss for predicting target from abstract variables
        z = torch.cat(self.validation_step_outputs.z_flat, dim=0)
        y = torch.cat(self.validation_step_outputs.y_flat, dim=0)

        max_data_points = 10000
        if z.shape[0] > max_data_points:
            # randomly sample max_data_points data points
            idx = torch.randperm(z.shape[0])[:max_data_points]
            z = z[idx]
            y = y[idx]
        z, y = z.cpu().numpy(), y.cpu().squeeze().numpy()

        y_train, y_val, z_train, z_val = train_test_split(
            y, z, test_size=0.2, random_state=42
        )
        # train random forest to predict y from z
        reg = RandomForestRegressor(
            n_estimators=100, max_depth=2, random_state=0, n_jobs=-1
        )
        reg.fit(z_train, y_train)
        y_pred = reg.predict(z_val)
        val_l2_loss = mean_squared_error(y_val, y_pred)
        self.validation_step_outputs.clear()  # release memory

        self.log("val_l2_loss", val_l2_loss, prog_bar=False)
        for name, param in self.named_parameters():
            self.log(name, param.mean(), prog_bar=False)

        # compare tau and omega maps to ground truth
        if self.reduction.tau_map_ground_truth is not None:
            tau_map_params = list(
                self.reduction.tau_map.segment_aggregation[0].parameters()
            )[0][0]
            tau_map_params = tau_map_params / tau_map_params.sum()  # normalize
            tau_map_params_gt = list(
                self.reduction.tau_map_ground_truth.segment_aggregation[0].parameters()
            )[0][0]
            tau_map_params_gt = tau_map_params_gt / tau_map_params_gt.sum()  # normalize
            tau_map_similarity = F.cosine_similarity(
                tau_map_params, tau_map_params_gt, dim=0
            )
            tau_map_similarity_norm = 1 - torch.abs(tau_map_similarity)
            self.log("val_tau_map_similarity", tau_map_similarity, prog_bar=False)
            self.log(
                "val_tau_map_similarity_norm", tau_map_similarity_norm, prog_bar=False
            )

        if self.reduction.omega_map_ground_truth is not None:
            omega_map_params = list(
                self.reduction.omega_map.segment_aggregation[0].parameters()
            )[0][0]
            omega_map_params = omega_map_params / omega_map_params.sum()
            omega_map_params_gt = list(
                self.reduction.omega_map_ground_truth.segment_aggregation[
                    0
                ].parameters()
            )[0][0]
            omega_map_params_gt = omega_map_params_gt / omega_map_params_gt.sum()
            omega_map_similarity = F.cosine_similarity(
                omega_map_params, omega_map_params_gt, dim=0
            )
            omega_map_similarity_norm = 1 - torch.abs(omega_map_similarity)
            self.log("val_omega_map_similarity", omega_map_similarity, prog_bar=False)
            self.log(
                "val_omega_map_similarity_norm",
                omega_map_similarity_norm,
                prog_bar=False,
            )

    def test_step(self, batch, batch_idx):
        pass

    def on_test_end(self) -> None:
        fig = None
        if self.reduction.low_level._get_name() == "LinearCausalModel":
            fig = plot_weights_two_branch(self.reduction)
        elif self.reduction.low_level._get_name() == "MassSpringCausalModel":
            fig = plot_weights_mass_spring(self.reduction)

        if fig is not None:
            filename = f"plot_{uuid.uuid4()}.png"
            fig.savefig(filename)
            # log to wandb from lightning logger
            self.logger.experiment.log(
                {
                    "tau_omega_weights": wandb.Image(filename),
                }
            )
            # delete plot
            os.remove(filename)

        for name, param in self.named_parameters():
            print(f"Parameter name: {name},\nParam: {param}\n")

    def configure_optimizers(self):
        config_dict = {}

        # only apply weight decay to the parameters of the omega and tau map
        optimizer_parameters = []
        for name, param in self.named_parameters():
            if "omega_map" in name.split(".") or "tau_map" in name.split("."):
                optimizer_parameters.append(
                    {"params": param, "weight_decay": self.weight_decay}
                )
            else:
                optimizer_parameters.append({"params": param})
        optimizer = torch.optim.Adam(optimizer_parameters, lr=self.lr)
        config_dict["optimizer"] = optimizer

        if self.lr_scheduler == "cosine":
            # cosine learning rate annealing
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.lr_min,
                verbose=True,
            )
            lr_scheduler_config = {
                "scheduler": lr_scheduler,
                "interval": "epoch",
            }
            config_dict["lr_scheduler"] = lr_scheduler_config
        elif self.lr_scheduler is None:
            return optimizer
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")
        return config_dict
