from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt

from ..causal_model import LowLevelCausalModel
from ..reduction import LinearReductionMap

CHAIN = "chain"
STAR = "star"
TWO_BRANCH = "two_branch"
SIMPLE_SPARSE = "simple_sparse"


class LinearCausalModel(LowLevelCausalModel):
    """
    Linear causal model with a sparse adjacency matrix. The adjacency matrix is specified by the
    parameter adj_matrix. The causal mechanism is specified by the parameter mechanism_param.
    """

    def __init__(
        self,
        n_vars: int,
        adj_matrix: Optional[np.ndarray] = None,
        graph_type: Optional[str] = None,
        intervention_mask: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(n_vars=n_vars, intervention_mask=intervention_mask)
        if adj_matrix is not None:
            assert adj_matrix.shape == (
                n_vars + 1,
                n_vars + 1,
            ), "adj_matrix shape mismatch"
        assert graph_type in [
            None,
            CHAIN,
            STAR,
            TWO_BRANCH,
            SIMPLE_SPARSE,
        ], f"graph type {graph_type} not supported"
        assert not (
            adj_matrix is not None and graph_type is not None
        ), "graph_type and adj_matrix cannot be specified at the same time"

        self.adj_matrix = (
            adj_matrix
            if adj_matrix is not None
            else self._sample_adjacency(n_vars, graph_type)
        )
        self.register_buffer(
            "adj_matrix_buffer", torch.from_numpy(self.adj_matrix).float()
        )
        self.graph_type = graph_type
        self.reduced_form = self._reduced_form(self.adj_matrix)

    def sample_exogenous_noise(self, batch_size) -> np.ndarray:
        return np.random.normal(size=(batch_size, self.n_vars + 1))

    def sample(
        self, intervention, batch_size, **kwargs
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from the causal model.
        """
        U = self.sample_exogenous_noise(batch_size)
        U[:, :-1] = U[:, :-1] + intervention
        X = U @ self.reduced_form

        data_X = X[:, :-1][np.newaxis, ...]
        data_I = intervention.repeat(batch_size, axis=0)[np.newaxis, ...]
        data_Y = X[:, [-1]][np.newaxis, ...]

        return data_X, data_I, data_Y

    def _sample_unmasked_intervention(self) -> np.ndarray:
        """
        Sample an intervention for all variables.
        """
        intervention = np.random.normal(size=(1, self.n_vars))
        return intervention

    def zero_intervention(self) -> np.ndarray:
        """
        Return a zero intervention.
        """
        return np.zeros((1, self.n_vars))

    def _sample_adjacency(
        self, n_vars: int, graph_type: Optional[str] = None
    ) -> np.ndarray:
        while True:
            adj_matrix = self._sample_adjacency_candidate(n_vars, graph_type)
            if self._check_rank(adj_matrix):
                break
        return adj_matrix

    @staticmethod
    def _sample_adjacency_candidate(
        n_vars: int, graph_type: Optional[str] = None
    ) -> np.ndarray:
        if graph_type == CHAIN:
            adj_matrix = np.zeros((n_vars + 1, n_vars + 1))
            adj_matrix[np.arange(n_vars), np.arange(1, n_vars + 1)] = np.random.uniform(
                -1, 1, size=n_vars
            )
        elif graph_type == STAR:
            adj_matrix = np.zeros((n_vars + 1, n_vars + 1))
            adj_matrix[:-1, -1] = np.random.uniform(-1, 1, size=n_vars)
        elif graph_type == TWO_BRANCH:
            len_branch_1 = (n_vars) // 2
            len_branch_2 = n_vars - len_branch_1
            adj_matrix = np.zeros((n_vars + 1, n_vars + 1))
            for i in range(len_branch_1 - 1):
                adj_matrix[i, i + 1] = np.random.uniform(-1, 1)

            # make sure edge to target has abs > 0.1
            adj_matrix[len_branch_1 - 1, -1] = np.random.uniform(-1, 1)
            while np.abs(adj_matrix[len_branch_1 - 1, -1]) < 0.1:
                adj_matrix[len_branch_1 - 1, -1] = np.random.uniform(-1, 1)

            for i in range(len_branch_2):
                from_idx = len_branch_1 + i
                to_idx = from_idx + 1
                adj_matrix[from_idx, to_idx] = np.random.uniform(-1, 1)

            # make sure edge to target has abs > 0.1
            while np.abs(adj_matrix[-2, -1]) < 0.1:
                adj_matrix[-2, -1] = np.random.uniform(-1, 1)
        elif graph_type is SIMPLE_SPARSE:
            adj_matrix = np.zeros((n_vars + 1, n_vars + 1))
            adj_matrix[0, -1] = np.random.uniform(-1, 1)
        else:
            # sample upper triangular adjacency matrix for DAG
            adj_matrix = np.random.uniform(-1, 1, size=(n_vars + 1, n_vars + 1))
            adj_matrix[np.tril_indices(n_vars + 1)] = 0  # set lower triangular to 0
        return adj_matrix

    @staticmethod
    def _reduced_form(adj_matrix: np.ndarray) -> np.ndarray:
        return np.linalg.inv(np.eye(adj_matrix.shape[0]) - adj_matrix)

    @staticmethod
    def _check_rank(adj_matrix: np.ndarray) -> bool:
        return (
            np.linalg.matrix_rank(np.eye(adj_matrix.shape[0]) - adj_matrix)
            == adj_matrix.shape[0]
        )


def ground_truth_solution(
    linear_causal_model: LinearCausalModel,
) -> tuple[LinearReductionMap, LinearReductionMap]:
    """
    Compute the ground truth solution for the linear causal model. This only implements the case
    where the target variable is the last variable and the high-level causal model has two
    variables.

    Parameters
    ----------
    linear_causal_model: LinearCausalModel
        The linear causal model.

    Returns
    -------
    tau_map_ground_truth: LinearReductionMap
        The ground truth solution for the tau map.
    omega_map_ground_truth: LinearReductionMap
        The ground truth solution for the omega map.
    """
    adj_matrix = linear_causal_model.adj_matrix

    # tau map
    tau_weights = np.concatenate((adj_matrix[:-1, -1], [0]))  # add zero bias term
    tau_weights = torch.from_numpy(np.atleast_2d(tau_weights)).float()
    n_vars = linear_causal_model.n_vars
    tau_map_ground_truth = LinearReductionMap(
        n_vars_low_level=n_vars, n_vars_high_level=1, trainable=False
    )
    tau_map_ground_truth.set_parameters(tau_weights)

    # omega map
    tau_omega_matrix = torch.from_numpy(
        np.linalg.inv(np.eye(n_vars) - adj_matrix[:-1, :-1])
    ).float()
    omega_weights = tau_omega_matrix @ tau_weights[:, :-1].flatten()
    omega_weights = torch.cat((omega_weights, torch.zeros(1))).unsqueeze(
        0
    )  # add zero bias term
    omega_map_ground_truth = LinearReductionMap(
        n_vars_low_level=n_vars, n_vars_high_level=1, trainable=False
    )
    omega_map_ground_truth.set_parameters(omega_weights)

    return tau_map_ground_truth, omega_map_ground_truth


def plot_weights_two_branch(reduction):
    if not (
        reduction.low_level.graph_type == "two_branch"
        and reduction.high_level.n_vars == 2
    ):
        return None
    tau1 = reduction.tau_map.get_weights(idx=0).cpu().numpy()
    tau2 = reduction.tau_map.get_weights(idx=1).cpu().numpy()
    omega1 = reduction.omega_map.get_weights(idx=0).cpu().numpy()
    omega2 = reduction.omega_map.get_weights(idx=1).cpu().numpy()
    tau1_norm = tau1.flatten() / np.sqrt(np.sum(tau1**2))
    tau2_norm = tau2.flatten() / np.sqrt(np.sum(tau2**2))
    omega1_norm = omega1.flatten() / np.sqrt(np.sum(omega1**2))
    omega2_norm = omega2.flatten() / np.sqrt(np.sum(omega2**2))
    # plot
    mm = 1 / (10 * 2.54)  # millimeters in inches
    SINGLE_COLUMN = 85 * mm
    color_list = [
        "#5790fc",
        "#f89c20",
        "#e42536",
        "#964a8b",
        "#9c9ca1",
        "#7a21dd",
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN, 0.4 * SINGLE_COLUMN), dpi=500)
    indices = np.arange(1, len(tau2_norm) + 1)
    plt.plot(
        indices,
        tau1_norm,
        label=r"$\mathbf{\tau}_1$",
        c=color_list[0],
        ls="-",
    )
    plt.plot(
        indices,
        omega1_norm,
        label=r"$\mathbf{\omega}_1$",
        c=color_list[1],
        ls="-",
    )
    plt.plot(
        indices,
        tau2_norm,
        label=r"$\mathbf{\tau}_2$",
        c=color_list[2],
        ls="--",
    )
    plt.plot(
        indices,
        omega2_norm,
        label=r"$\mathbf{\omega}_2$",
        c=color_list[3],
        ls="--",
    )
    plt.legend()
    plt.xlabel("Variable index")
    plt.ylabel("Normalised weight")
    plt.xlim(indices.min(), indices.max())
    # ax = plt.gca()
    plt.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        labelbottom=True,
        right=False,
        left=True,
        labelleft=True,
    )
    return fig
