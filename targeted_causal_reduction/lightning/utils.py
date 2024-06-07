from dataclasses import dataclass

import torch


def batch_cov(input: torch.Tensor) -> torch.Tensor:
    """
    Compute batched covariance matrices for a given input tensor.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor of shape (batch_size, num_samples, num_variables).

    Returns
    -------
    torch.Tensor
        Batched covariance matrices of shape (batch_size, 1, num_variables, num_variables).

    References
    ----------
    https://stackoverflow.com/questions/71357619/how-do-i-compute-batched-sample-covariance-in-pytorch
    """

    # Step 1: Calculate the mean along the second dimension
    mean_input = torch.mean(
        input, dim=1, keepdim=True
    )  # shape: [batch_size, 1, num_variables]

    # Step 2: Subtract the mean from 'input'
    input_centered = (
        input - mean_input
    )  # shape: [batch_size, num_samples, num_variables]

    # Step 3: Compute the covariance matrix using batched matrix multiplication
    # Transpose input_centered and use torch.bmm to calculate the covariance
    cov_matrix = torch.bmm(
        input_centered.transpose(1, 2), input_centered
    )  # shape: [batch_size, num_variables, num_variables]

    # Step 4: Normalize the covariance matrix
    # Divide by (n-1) for unbiased estimate
    n = input.size(1)  # number of samples in the second dimension
    cov_matrix /= n - 1
    num_vars = input.size(2)

    # Reshape the covariance matrix to have shape (batch_size, 1, num_variables, num_variables)
    cov_matrix = cov_matrix.view(-1, 1, num_vars, num_vars)
    return cov_matrix


@dataclass
class ValidationStepOutputs:
    z_flat: list[torch.Tensor]
    r_flat: list[torch.Tensor]
    y_flat: list[torch.Tensor]
    loss: list[torch.Tensor]

    def clear(self) -> None:
        self.z_flat = []
        self.r_flat = []
        self.y_flat = []
        self.loss = []
