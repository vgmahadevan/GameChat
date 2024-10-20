"""Configurations for the MPC controller."""

import numpy as np
from enum import Enum, auto

class DynamicsModel(Enum):
    SINGLE_INTEGRATOR = auto()
    DOUBLE_INTEGRATOR = auto()

# Liveness parameters.
liveliness = True
liveness_threshold = 0.3
plot_rate = 1
# dynamics = DynamicsModel.SINGLE_INTEGRATOR
dynamics = DynamicsModel.DOUBLE_INTEGRATOR

if dynamics == DynamicsModel.SINGLE_INTEGRATOR:
    num_states = 3 # (x, y, theta)
    num_controls = 2 # (v, omega)
else:
    num_states = 4 # (x, y, theta, v)
    num_controls = 2 # (a, omega)

Ts = 0.2                                   # Sampling time [s]
T_horizon = 4                              # Prediction horizon time steps

obs_gamma = 0.1                            # CBF parameter in [0,1]
liveliness_gamma = 0.1                     # CBF parameter in [0,1]
safety_dist = 0.03                         # Safety distance
agent_radius = 0.1                         # Robot radius (for obstacle avoidance)

# Actuator limits
v_limit = 0.30                             # Linear velocity limit
omega_limit = 1.0                          # Angular velocity limit
accel_limit = 0.5

zeta = 3.0

# ------------------------------------------------------------------------------
COST_MATRICES = {
    DynamicsModel.SINGLE_INTEGRATOR: {
        "Q": np.diag([15, 15, 0.005]),  # State cost matrix DOORWAY
        # "Q": np.diag([100, 100, 11]), # State cost matrix INTERSECTION
        "R": np.array([3, 1.5]),                  # Controls cost matrix
    },
    DynamicsModel.DOUBLE_INTEGRATOR: {
        "Q": np.diag([15, 15, 0.005, 3]),  # State cost matrix DOORWAY
        # "Q": np.diag([100, 100, 11, 3]), # State cost matrix INTERSECTION
        "R": np.array([0.5, 10.0]),                  # Controls cost matrix
    }
}
