from abc import ABC, abstractmethod

import numpy as np


class Simulator(ABC):
    """
    Base class for simulators. It defines the interface for running a simulation under a shift intervention and given
    initial conditions. It also provides a method to sample initial conditions.
    """
    @abstractmethod
    def run(self, y0, delta_y, **kwargs) -> np.ndarray:
        """
        Run the simulator and output simulation data.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_initial_conditions(self) -> np.ndarray:
        """
        Sample initial conditions from the distribution over initial conditions.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def time_steps(self) -> np.ndarray:
        """
        Return the time points of the simulation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError
