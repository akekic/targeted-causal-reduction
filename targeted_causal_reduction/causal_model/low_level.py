from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from torch import nn


class LowLevelCausalModel(ABC, nn.Module):
    """
    Base class for low-level causal models. It defines the interface for sampling interventions as well as
    samples of the causal model under the interventions.

    Attributes
    ----------
    n_vars : int
        The number of variables in the causal model, excluding the target variable.
    intervention_mask : np.ndarray
        The mask for the intervention. It is a binary mask of shape (1, n_vars). If the mask is 0, the variable is
        not intervened upon; if the mask is 1, the variable is intervened upon.
    """

    def __init__(
        self,
        n_vars: int,  # excluding target variable
        intervention_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Parameters
        ----------
        n_vars: int
            The number of variables in the causal model, excluding the target variable.
        intervention_mask: Optional[np.ndarray]
            The mask for the intervention. If None, all non-target variables are intervened upon.
        """
        super().__init__()
        self.n_vars = n_vars
        self.intervention_mask = self._set_intervention_mask(intervention_mask)

    @abstractmethod
    def sample(
        self, intervention: np.ndarray, batch_size: int, **kwargs: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample from the causal model.

        Parameters
        ----------
        intervention : np.ndarray
            The shift intervention to apply. The shape is (1, n_vars).
        batch_size : int
            The number of samples to generate.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            The features X, the intervention I and the target Y. The shape of X and I is (1, batch_size, n_vars);
            the shape of Y is (1, batch_size, 1).
        """
        raise NotImplementedError

    @abstractmethod
    def _sample_unmasked_intervention(self) -> np.ndarray:
        """
        Sample an intervention for all variables. Some variables may be masked out by the intervention mask.

        Returns
        -------
        np.ndarray
            The unmasked intervention of shape (1, n_vars).
        """
        raise NotImplementedError

    def sample_intervention(self) -> np.ndarray:
        """
        Sample an intervention.

        Returns
        -------
        np.ndarray
            The masked intervention of shape (1, n_vars).
        """
        unmasked_intervention = self._sample_unmasked_intervention()
        return unmasked_intervention * self.intervention_mask

    @abstractmethod
    def zero_intervention(self) -> np.ndarray:
        """
        Return a zero intervention.

        Returns
        -------
        np.ndarray
            The zero intervention of shape (1, n_vars).
        """
        raise NotImplementedError

    def _set_intervention_mask(
        self, intervention_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Parameters
        ----------
        intervention_mask: Optional[np.ndarray]
            The mask for the intervention. If None, all non-target variables are intervened upon.

        Returns
        -------
        np.ndarray
            The intervention mask of shape (1, n_vars).
        """
        assert intervention_mask is None or intervention_mask.shape == (
            1,
            self.n_vars,
        ), "intervention_mask shape mismatch"
        intervention_mask_out = (
            intervention_mask
            if intervention_mask is not None
            else np.ones((1, self.n_vars))
        )  # mask given as parameter
        return intervention_mask_out

    def attributes(self) -> dict:
        """
        Return the attributes of the low-level causal model which uniquely identify the data it produces.
        This is used to store metadata about the data in the HDF5 file. It has no effect for in-memory datasets.
        """
        raise NotImplementedError
