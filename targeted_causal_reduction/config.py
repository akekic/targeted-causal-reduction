import numpy as np

from targeted_causal_reduction.data_generators import (
    DoubleWellCausalModel,
    TripleWellCausalModel,
    LinearCausalModel,
    MassSpringCausalModel,
)
from targeted_causal_reduction.data_generators.linear import (
    SIMPLE_SPARSE,
    CHAIN,
    STAR,
    TWO_BRANCH,
)

CLASS = "class"
CLASS_ARGS = "class_args"
SIM_ARGS = "sim_args"
CHAIN_INTERVENTION_MASK = np.ones((1, 9))
CHAIN_INTERVENTION_MASK[:, -2:-1] = 0.0  # TODO: what to do with this?

LOW_LEVEL_MODELS = {
    "double_well": {
        CLASS: DoubleWellCausalModel,
        CLASS_ARGS: {"n_time_steps": 101, "total_time": 10.0},
        SIM_ARGS: {},
    },
    "triple_well": {
        CLASS: TripleWellCausalModel,
        CLASS_ARGS: {"n_time_steps": 101, "total_time": 40.0},
        SIM_ARGS: {},
    },
    SIMPLE_SPARSE: {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {"n_vars": 9, "graph_type": SIMPLE_SPARSE},
        SIM_ARGS: {},
    },
    "linear": {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {"n_vars": 9},
        SIM_ARGS: {},
    },
    CHAIN: {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {"n_vars": 9, "graph_type": CHAIN},
        SIM_ARGS: {},
    },
    "chain_masked": {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {
            "n_vars": 9,
            "graph_type": CHAIN,
            "intervention_mask": CHAIN_INTERVENTION_MASK,
        },
        SIM_ARGS: {},
    },
    STAR: {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {"n_vars": 9, "graph_type": STAR},
        SIM_ARGS: {},
    },
    TWO_BRANCH: {
        CLASS: LinearCausalModel,
        CLASS_ARGS: {"n_vars": 9, "graph_type": TWO_BRANCH},
        SIM_ARGS: {},
    },
    "mass_spring": {
        CLASS: MassSpringCausalModel,
        CLASS_ARGS: {
            "n_masses": 4,
            "n_time_steps": 21,
            "group": False,
            "total_time": 100,
            "ic_noise": 1e-2,
            "intervention_noise": 5e-3,
            "k": 0.001,
            "c": 0.0,
            "rest_length": 1.0,
            "m": None,
            "target_direction": (1, 1),
        },
        SIM_ARGS: {},
    },
    "mass_spring_grouped": {
        CLASS: MassSpringCausalModel,
        CLASS_ARGS: {
            "n_masses": 8,
            "n_time_steps": 21,
            "group": True,
            "total_time": 100,
            "ic_noise": 1e-2,
            "intervention_noise": 5e-3,
            "k": 0.001,
            "c": 0.0,
            "rest_length": 1.0,
            "m": 1.0,
            "intervention_sparsity": 0.25,
            "target_direction": (1, 0),
        },
        SIM_ARGS: {},
    },
    "mass_no_spring": {
        CLASS: MassSpringCausalModel,
        CLASS_ARGS: {
            "n_masses": 4,
            "n_time_steps": 21,
            "group": False,
            "total_time": 100,
            "ic_noise": 1e-2,
            "intervention_noise": 5e-3,
            "k": 0.0,
            "c": 0.0,
            "rest_length": 1.0,
            "m": None,
        },
        SIM_ARGS: {},
    },
}
