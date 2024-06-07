import torch
from torch import nn


class LinearReductionMap(nn.Module):

    """
    A linear reduction map takes either (a) the variables or (b) the interventions of a low-level causal model and
    returns the corresponding variables or interventions of a high-level causal model. It is a linear map that
    aggregates the low-level variables or interventions into the high-level variables or interventions.

    Attributes
    ----------
    n_vars_low_level: int
        The number of low-level variables (excluding the target variable).
    n_vars_high_level: int
        The number of high-level variables (excluding the target variable).
    abstract_var_dim: int
        The dimension of the abstract variables. Defaults to 1.
    trainable: bool
        Whether the parameters of the segmentation can be trained. If False, the parameters are registered as
        buffers and are not updated via backpropagation. The parameters can still be set manually via
        `set_parameters`. Defaults to True.
    segment_aggregation: nn.ModuleList
        A list of nn.Linear modules that aggregate the low-level variables into the high-level variables. There is one
        nn.Linear module for each high-level variable.
    """

    def __init__(
        self,
        n_vars_low_level: int,
        n_vars_high_level: int,
        abstract_var_dim: int = 1,
        trainable=True,
    ) -> None:
        """
        Initialize a tau map.

        Parameters
        ----------
        n_vars_low_level: int
            The number of low-level variables (excluding the target variable).
        n_vars_high_level: int
            The number of abstract variables (excluding the target variable).
        abstract_var_dim: int
            The dimension of the abstract variables. Defaults to 1.
        trainable: bool
            Whether the parameters of the segmentation can be trained. If False, the parameters are registered as
            buffers and are not updated via backpropagation. The parameters can still be set manually via
            `set_parameters`. Defaults to True.
        """
        super().__init__()
        self.n_vars_low_level = n_vars_low_level
        self.n_abstract_vars = n_vars_high_level
        self.abstract_var_dim = abstract_var_dim
        self.trainable = trainable

        self.segment_aggregation = nn.ModuleList(
            [
                nn.Linear(n_vars_low_level, abstract_var_dim, bias=False)
                for _ in range(n_vars_high_level)
            ]
        )
        if not trainable:
            # register as buffers so that they are not registered as parameters
            for i in range(n_vars_high_level):
                self.register_buffer(
                    f"segment_aggregation_{i}_weight",
                    self.segment_aggregation[i].weight,
                )

    def forward(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        """
        Forward pass of the reduction map. It aggregates low-level variables or interventions into high-level variables
        or interventions.

        Parameters
        ----------
        x: torch.Tensor
            A tensor of shape (batch_size, *low_level_shape) containing the low-level variables or interventions.

        Returns
        -------
        z: torch.Tensor
            A tensor of shape (batch_size, n_abstract_vars) containing the abstract variables or interventions.
        """
        return self.segment_aggregation[idx](x)

    def set_parameters(self, parameters):
        assert (
            len(parameters) == self.n_abstract_vars
        ), "number of parameters does not match number of abstract variables minus target variable"
        for i, parameter in enumerate(parameters):
            weight = parameter[:-1].unsqueeze(0)
            bias = parameter[-1].unsqueeze(0)
            assert (
                weight.shape == self.segment_aggregation[i].weight.shape
            ), f"weight shape mismatch for segment {i}: {weight.shape} != {self.segment_aggregation[i].weight.shape}"

            self.segment_aggregation[i].weight = torch.nn.Parameter(weight)
            self.segment_aggregation[i].bias = torch.nn.Parameter(bias)

    def get_weights(self, idx: int):
        return self.segment_aggregation[idx].weight.squeeze()
