from typing import Optional, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

from ..causal_model import LowLevelCausalModel
from ..simulator import Simulator


def create_adjacency_matrix(n, side_length, group):
    """
    Create an adjacency matrix for a grid of n masses arranged in a square layout.
    """
    if group:
        A = np.zeros((n, n))
        A[: n // 2, : n // 2] = create_submatrix(n // 2, side_length)  # First group
        A[n // 2 :, n // 2 :] = create_submatrix(n // 2, side_length)  # Second group
    else:
        A = create_submatrix(n, side_length)

    return A


def create_submatrix(n, side_length):
    """
    Create an adjacency matrix for a grid of n masses arranged in a square layout.
    """
    A = np.zeros((n, n))
    for i in range(n):
        # Connect to the mass on the left and right
        if i % side_length != 0:
            A[i, i - 1] = 1
        if (i + 1) % side_length != 0:
            A[i, i + 1] = 1

        # Connect to the mass above and below
        if i - side_length >= 0:
            A[i, i - side_length] = 1
        if i + side_length < n:
            A[i, i + side_length] = 1

    return A


# Differential equation
def equation(t, y, A, rest_length, k, c, m):
    n = int(len(y) / 4)
    x = y[: 2 * n].reshape(n, 2)
    v = y[2 * n :].reshape(n, 2)
    derivatives = np.zeros_like(y)

    # Calculate forces using vectorized operations
    for i in range(n):
        connected = A[i, :] == 1
        u = x[i] - x[connected]
        distance = np.linalg.norm(u, axis=1, keepdims=True)
        spring_force = -k * (u - u / distance * rest_length)
        damping_force = -c * (v[i] - v[connected])
        net_force = np.sum(spring_force + damping_force, axis=0)
        acceleration = net_force / m[i]
        derivatives[2 * n + 2 * i : 2 * n + 2 * i + 2] = acceleration
        derivatives[i * 2 : i * 2 + 2] = v[i]

    return derivatives


class MassSpringSimulator(Simulator):
    def __init__(
        self,
        n_masses=9,
        n_time_steps=101,
        total_time=100,
        group=False,
        ic_noise=1e-3,
        k=0.01,
        c=0.0,
        rest_length=1.0,
        m: Optional[Union[float, np.ndarray]] = None,
    ):
        super().__init__()
        if group:
            assert (
                n_masses % 2 == 0 and np.sqrt(n_masses / 2) % 1 == 0
            ), "n_masses must be a square number"
        else:
            assert np.sqrt(n_masses) % 1 == 0, "n_masses must be a square number"
        self.n_masses = n_masses
        self.n_time_steps = n_time_steps
        self.total_time = total_time
        self.group = group
        self.ic_noise = ic_noise
        self.side_length = (
            int(np.sqrt(n_masses // 2)) if group else int(np.sqrt(n_masses))
        )
        self.A = create_adjacency_matrix(n_masses, self.side_length, group)
        self.k = k  # Spring constant
        self.c = c  # Damping constant
        self.rest_length = rest_length  # Rest length of the spring
        self.m = self._set_mass(m, n_masses)  # Mass of the masses

    @property
    def time_steps(self):
        return np.linspace(0, self.total_time, self.n_time_steps)

    def run(self, y0, delta_y, **kwargs) -> np.ndarray:
        """
        Run the simulator and output simulation data.
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
            time_steps_part = self.time_steps[start_idx : end_idx + 1]

            sol_part = odeint(
                equation,
                y0_part,
                time_steps_part,
                args=(self.A, self.rest_length, self.k, self.c, self.m),
                tfirst=True,
                atol=1e-4,
                rtol=1e-3,
            )
            if not start_idx == 0:
                sol_part = sol_part[1:, :]
            sol.append(sol_part)
        sol = np.concatenate(sol, axis=0)
        return sol

    def shape(self):
        return self.n_masses

    def create_initial_positions(self):
        """
        Create initial positions for n masses arranged in a grid.
        If group is True, create two separate groups.
        """
        positions = np.zeros((self.n_masses, 2))

        for i in range(self.n_masses):
            positions[i] = [
                (i % self.side_length) * self.rest_length,
                (i // self.side_length) * self.rest_length,
            ]

            # Offset the second group
            if self.group and i >= self.n_masses // 2:
                positions[i] += [
                    self.side_length * self.rest_length * 1.5,
                    0,
                ]  # Adjust the offset as needed

        # add a random offset to the initial positions
        offset_x = np.random.normal() * self.rest_length * 10
        offset_y = np.random.normal() * self.rest_length * 10
        positions[:, 0] += offset_x
        positions[:, 1] += offset_y

        return positions

    def sample_initial_conditions(self):
        """
        Sample initial conditions from the distribution over initial conditions.
        """
        sqrt_n = int(np.sqrt(self.n_masses))
        # Initial conditions
        x0 = self.create_initial_positions()  # positions
        v0 = self.ic_noise * np.random.normal(size=x0.shape)  # velocities
        y0 = np.hstack((x0.flatten(), v0.flatten()))
        return y0

    def _set_mass(self, m, n_masses):
        if isinstance(m, float):
            return np.ones(n_masses) * m
        elif isinstance(m, np.ndarray):
            assert m.shape == (n_masses,)
            return m
        elif m is None:
            return np.linspace(0.5, 1.5, n_masses)


class MassSpringCausalModel(LowLevelCausalModel):
    def __init__(
        self,
        n_masses=9,
        n_time_steps=101,
        total_time=100,
        group=False,
        intervention_noise=1e-4,
        ic_noise=1e-3,
        k=0.01,
        c=0.0,
        rest_length=1.0,
        m: Optional[Union[float, np.ndarray]] = None,
        intervention_sparsity: float = 1.0,
        remove_position: bool = True,
        target_direction: tuple[float, float] = (1.0, 1.0),
    ):
        # assert np.sqrt(n_masses) % 1 == 0, "n_masses must be a square number"
        n_vars = (2 if remove_position else 4) * n_masses * (n_time_steps - 1)
        super().__init__(n_vars)
        self.sim = MassSpringSimulator(
            n_masses=n_masses,
            n_time_steps=n_time_steps,
            total_time=total_time,
            group=group,
            ic_noise=ic_noise,
            k=k,
            c=c,
            rest_length=rest_length,
            m=m,
        )
        self.intervention_noise = intervention_noise
        self.intervention_sparsity = intervention_sparsity
        self.remove_position = remove_position
        self.target_direction = np.array(target_direction) / np.linalg.norm(
            target_direction
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
            I_out = (
                intervention_sim[:-1, 2 * self.sim.n_masses :].reshape((1, -1))
                if self.remove_position
                else intervention
            )

            if i == 0:
                data_X = X_out
                data_I = I_out
                data_Y = Y_out
            else:
                data_X = np.concatenate((data_X, X_out), axis=0)
                data_I = np.concatenate((data_I, I_out), axis=0)
                data_Y = np.concatenate((data_Y, Y_out), axis=0)

        return data_X[np.newaxis, ...], data_I[np.newaxis, ...], data_Y[np.newaxis, ...]

    def _sample_unmasked_intervention(self) -> np.ndarray:
        shape = (self.sim.n_time_steps - 1, self.sim.n_masses * 2)
        delta_vel = self.intervention_noise * np.random.normal(size=shape)
        if self.intervention_sparsity < 1.0:
            mask = np.random.binomial(1, self.intervention_sparsity, size=shape).astype(
                bool
            )
            delta_vel[~mask] = 0.0
        if self.remove_position:
            delta_y = delta_vel.reshape((1, -1))
        else:
            delta_pos = np.zeros_like(delta_vel)
            delta_y = np.hstack((delta_pos, delta_vel)).reshape((1, -1))
        return delta_y

    def _to_sim_intervention(self, intervention: np.ndarray) -> np.ndarray:
        """
        Convert an intervention in the causal model to an intervention in the simulator.
        """
        intervention_sim = intervention.reshape((self.sim.n_time_steps - 1, -1))
        if self.remove_position:
            # add intervention for positions
            delta_pos = np.zeros_like(intervention_sim)
            intervention_sim = np.hstack((delta_pos, intervention_sim))
        # add a zero intervention for the last time step
        intervention_sim = np.concatenate(
            (intervention_sim, np.zeros((1, self.sim.n_masses * 4))), axis=0
        )
        return intervention_sim

    def zero_intervention(self) -> np.ndarray:
        return np.zeros((1, self.n_vars))

    def _unpack_sim(self, sol):
        X = (
            sol[:-1, 2 * self.sim.n_masses :] if self.remove_position else sol[:-1, ...]
        ).reshape((1, -1))
        # target is average velocity of last time step
        vel = sol[-1, self.sim.n_masses * 2 :].reshape((self.sim.n_masses, 2))
        Y_x = np.average(vel, weights=self.sim.m, axis=0)[0].reshape(
            (1, -1)
        )  # c.o.m. velocity in x-direction
        Y_y = np.average(vel, weights=self.sim.m, axis=0)[1].reshape(
            (1, -1)
        )  # c.o.m. velocity in y-direction
        # c.o.m. velocity in (1,1) direction
        Y = np.dot(self.target_direction, np.array([Y_x.flatten(), Y_y.flatten()]))[
            np.newaxis, :
        ]
        return X, Y

    def attributes(self) -> dict:
        return {
            "n_masses": self.sim.n_masses,
            "n_time_steps": self.sim.n_time_steps,
            "total_time": self.sim.total_time,
            "group": self.sim.group,
            "intervention_noise": self.intervention_noise,
            "ic_noise": self.sim.ic_noise,
            "k": self.sim.k,
            "c": self.sim.c,
            "rest_length": self.sim.rest_length,
            "m": self.sim.m,
            "intervention_sparsity": self.intervention_sparsity,
            "remove_position": self.remove_position,
            "target_direction": self.target_direction,
        }


def plot_weights_mass_spring(reduction):
    SINGLE_COLUMN = 85 * 1 / (10 * 2.54)
    fig = plt.figure(
        figsize=(
            2 * SINGLE_COLUMN,
            0.6 * reduction.high_level.n_vars * SINGLE_COLUMN,
        ),
        dpi=500,
    )
    gs = fig.add_gridspec(reduction.high_level.n_vars + 1, 2)
    axes = []
    for i in range(reduction.high_level.n_vars):
        axes_row = []
        for j in range(2):
            if i == 0:
                ax = plt.subplot(gs[i, j])
            else:
                ax = plt.subplot(gs[i, j], sharex=axes[0][j])
            axes_row.append(ax)
        axes.append(axes_row)
    axes = np.array(axes)
    axis_mechanism = plt.subplot(gs[-1, :])

    color_list = plt.cm.viridis(np.linspace(0, 1, reduction.low_level.sim.n_masses))
    alpha = 0.5
    remove_position = reduction.low_level.remove_position
    shape = (
        reduction.low_level.sim.n_time_steps - 1,
        (1 if remove_position else 2) * reduction.low_level.sim.n_masses,
        2,
    )
    for idx in range(reduction.high_level.n_vars):
        tau = reduction.tau_map.get_weights(idx=idx).cpu().numpy()
        omega = reduction.omega_map.get_weights(idx=idx).cpu().numpy()
        tau = tau.reshape(shape)
        omega = omega.reshape(shape)
        # plot
        for mass_idx in range(shape[1]):
            c = color_list[mass_idx]
            axes[idx, 0].plot(
                tau[:, mass_idx, 0],
                label=rf"$var{idx}x$",
                c=c,
                alpha=alpha,
                ls="-",
            )
            axes[idx, 0].plot(
                tau[:, mass_idx, 1],
                label=rf"$var{idx}y$",
                c=c,
                alpha=alpha,
                ls=":",
            )
            axes[idx, 1].plot(
                omega[:, mass_idx, 0],
                label=rf"$var{idx}x$",
                c=c,
                alpha=alpha,
                ls="-",
            )
            axes[idx, 1].plot(
                omega[:, mass_idx, 1],
                label=rf"$var{idx}y$",
                c=c,
                alpha=alpha,
                ls=":",
            )
            # Move the y-axis ticks to the right
            axes[idx, 1].yaxis.set_ticks_position("right")

            # Move the y-axis label to the right
            axes[idx, 1].yaxis.set_label_position("right")

            # only show x axis tick labels for the bottom row
            if idx == reduction.high_level.n_vars - 1:
                axes[idx, 0].set_xlabel("Time step")
                axes[idx, 1].set_xlabel("Time step")
            else:
                axes[idx, 0].set_xticks([])
                axes[idx, 1].set_xticks([])

    axes[0, 0].set_title(r"$\mathbf{\tau}$-parameters")
    axes[0, 1].set_title(r"$\mathbf{\omega}$-parameters")

    mech_weights = reduction.high_level.mlp[0].weight.squeeze().cpu().numpy()
    # bar plot of absolute mechanism weights
    axis_mechanism.bar(
        np.arange(mech_weights.shape[0]),
        np.abs(mech_weights),
        color="gray",
    )
    axis_mechanism.set_xticks(np.arange(mech_weights.shape[0]))
    axis_mechanism.set_xlabel("Variable index")
    axis_mechanism.set_ylabel("Absolute weight")
    return fig
