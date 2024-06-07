from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import odeint

from ...causal_model import LowLevelCausalModel
from ...simulator import Simulator


class NWell(Simulator, ABC):
    def __init__(self, n_time_steps=11, total_time=40.0):
        super().__init__()
        self.n_time_steps = n_time_steps
        self.total_time = total_time

    def run(self, y0, delta_y, k=5, m=1.0) -> np.ndarray:
        """
        Run the simulator and output simulation data.

        Parameters
        ----------
        y0: array-like, shape (n_variables,)
            Initial conditions.
        delta_y: array-like, shape (n_time_steps, n_variables)
            Shift intervention on the state vector [x, v] for each time step.
        k: float
            Friction coefficient.
        m: float
            Mass.

        Returns
        -------
        sol: array-like
            Simulation data.
        """
        # divide time steps into parts where no shift intervention is applied
        # then integrate between unintervened time steps
        intervention_time_steps = np.argwhere(delta_y.sum(axis=1) != 0).flatten()
        sol = []
        for i in range(len(intervention_time_steps) + 1):
            if len(intervention_time_steps) == 0:
                # no intervention
                start_idx = 0
                end_idx = len(self.time_steps)
            elif i == 0:
                start_idx = 0
                end_idx = intervention_time_steps[i]
                if end_idx == 0:
                    continue
            elif i == len(intervention_time_steps):
                start_idx = intervention_time_steps[i - 1]
                end_idx = len(self.time_steps)
            else:
                start_idx = intervention_time_steps[i - 1]
                end_idx = intervention_time_steps[i]

            y0_part = y0 if len(sol) == 0 else sol[-1][-1, :]
            y0_part += delta_y[start_idx, :]

            y0_part[1] += 0.2 * np.random.normal()
            time_steps_part = self.time_steps[start_idx : end_idx + 1]
            sol_part = odeint(
                self.ode_func,
                y0_part,
                time_steps_part,
                tfirst=True,
                Dfun=self.jacobian_func,
                col_deriv=True,
                args=(k, m),
                atol=1e-4,
                rtol=1e-3,
            )
            if not start_idx == 0:
                sol_part = sol_part[1:, :]
            sol.append(sol_part)
        sol = np.concatenate(sol, axis=0)
        return sol

    @abstractmethod
    def sample_initial_conditions(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def time_steps(self) -> np.ndarray:
        """
        Return the time points of the simulation.
        """
        return np.linspace(0, self.total_time, self.n_time_steps)

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.time_steps), 2)


class NWellCausalModel(LowLevelCausalModel):
    sim_class: NWell = None
    target_index = slice(None), 0

    def __init__(self, n_time_steps, total_time=40.0, intervention_mask=None, **kwargs):
        self.sim = self.sim_class(n_time_steps, total_time)
        super().__init__(
            n_vars=2 * (len(self.sim.time_steps) - 1),
            intervention_mask=intervention_mask,
        )

    def sample(self, intervention, batch_size, **kwargs):
        """
        Sample from the causal model.
        """
        global data_X, data_I, data_Y

        for i in range(batch_size):
            y0 = self.sim.sample_initial_conditions()
            intervention_sim = self._to_sim_intervention(intervention)
            X_out, Y_out = self._unpack_sim(self.sim.run(y0, intervention_sim))
            I_out = intervention

            if i == 0:
                data_X = X_out
                data_I = I_out
                data_Y = Y_out
            else:
                data_X = np.concatenate((data_X, X_out), axis=0)
                data_I = np.concatenate((data_I, I_out), axis=0)
                data_Y = np.concatenate((data_Y, Y_out), axis=0)

        return data_X[np.newaxis, ...], data_I[np.newaxis, ...], data_Y[np.newaxis, ...]

    def zero_intervention(self):
        """
        Return a zero intervention.
        """
        zero_intervention = np.zeros(self.sim.shape)
        return zero_intervention[:-1, :].reshape((1, -1))

    def _unpack_sim(self, sol):
        assert sol.shape == self.sim.shape, "unexpected input shape"

        y = np.atleast_1d(sol[-1, 0])  # target
        x = sol[:-1, :]
        return x.reshape((1, -1)), y.reshape((1, -1))

    def _to_sim_intervention(self, intervention):
        intervention_sim = intervention.reshape((len(self.sim.time_steps) - 1, 2))
        # add target to interventions
        intervention_sim = np.concatenate((intervention_sim, np.zeros((1, 2))), axis=0)
        return intervention_sim
