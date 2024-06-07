import torch
from torch import nn


class HighLevelCausalModel(nn.Module):
    """
    Linear Gaussian high-level causal model with `n_vars` variables causes 'z' and one effect variable
    (called target variable) 'y'.

    Attributes
    ----------
    n_vars : int
        The number of variables in the causal model, excluding the target variable; i.e., the number of causes.
    mu_z : nn.Parameter
        The mean of the unintervened cause variables 'z' of shape (n_vars,).
    sigma_z : nn.Parameter
        The standard deviation of the unintervened cause variables 'z' of shape (n_vars,).
    sigma_y_given_z : nn.Parameter
        The standard deviation of the target variable 'y' given the cause variables 'z'.
    mlp : nn.Module
        The linear layer to model the causal mechanism.

    mu_z, sigma_z, sigma_y_given_z and the weights of the mlp are learnable parameters.
    """

    def __init__(self, n_vars: int) -> None:
        super().__init__()
        self.n_vars = n_vars

        # register parameters for mu_z, sigma_z and sigma_y_given_z
        self.mu_z = nn.Parameter(torch.zeros(n_vars))
        self.sigma_z = nn.Parameter(torch.ones(n_vars))
        self.sigma_y_given_z = nn.Parameter(torch.ones(1))

        self.mlp = nn.Sequential(nn.Linear(n_vars, 1))

    def causal_mechanism(self, z):
        return self.mlp(z)
