import numpy as np
from scipy.integrate import odeint

from .base import NWell, NWellCausalModel


def triple_well_ode(t, y, k, m) -> list[float]:
    """
    ODE for the triple well system.
    Parameters
    ----------
    t: array-like
        Current time point.
    y: array-like
        [x, v]
    k: float
        Friction coefficient.
    m: float
        Mass.

    Returns
    -------
    dydt: list[float]
        [dxdt, dvdt]
    """
    x, v = y
    dydt = [v, (-1.0 / m) * (k * v + 6 * x**5 - 8 * x**3 + 2 * x)]
    return dydt


def triple_well_jacobian(t, y, k, m) -> np.ndarray:
    """
    Jacobian of the triple well system.
    Parameters
    ----------
    t: array-like
        Current time point.
    y: array-like
        [x, v]
    k: float
        Friction coefficient.
    m: float
        Mass.

    Returns
    -------
    jac: array-like
        Jacobian matrix.
    """
    x, v = y
    jac = [
        [0, 1],
        [(-1.0 / m) * (30 * x**4 - 24 * x**2 + 2), (-k / m)],
    ]
    return np.array(jac)


class TripleWell(NWell):
    _ode_func = triple_well_ode
    _jacobian_func = triple_well_jacobian

    def sample_initial_conditions(self) -> np.ndarray:
        coin = np.random.binomial(1, 0.5)
        dx = 0.01
        x_right = np.random.uniform(-2.1051 - dx, -2.1051 + dx)
        x_left = np.random.uniform(-1.6473 - dx, -1.6473 + dx)
        x0 = x_right if coin else x_left
        v0 = 0
        return np.array([x0, v0])

    def run(self, y0, delta_y, k=4.0, m=1.0) -> np.ndarray:
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

            y0_part[1] += 0.1 * np.random.normal()
            time_steps_part = self.time_steps[start_idx : end_idx + 1]
            sol_part = odeint(
                triple_well_ode,
                y0_part,
                time_steps_part,
                tfirst=True,
                Dfun=triple_well_jacobian,
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


class TripleWellCausalModel(NWellCausalModel):
    sim_class = TripleWell

    def _sample_unmasked_intervention(self) -> np.ndarray:
        delta_y = 0.2 * np.random.normal(size=self.sim.shape)
        delta_y = delta_y[:-1, :]  # remove target from interventions
        delta_y[:, 0] = 0.0  # no intervention on the position
        return delta_y.reshape((1, -1))

    def attributes(self):
        """
        Return the attributes of the causal model which uniquely identify the data it produces.
        """
        return {
            "class": self.__class__.__name__,
            "n_time_steps": self.sim.n_time_steps,
            "total_time": self.sim.total_time,
        }
