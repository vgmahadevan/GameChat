"""Configurations for the MPC controller."""

import numpy as np

sim_time = 4                             # Total simulation time steps
Ts = 0.1                                  # Sampling time [s]
T_horizon = 3                         # Prediction horizon time steps

gamma = 0.1                                # CBF parameter in [0,1]
safety_dist = 0.03                         # Safety distance

# Actuator limits
v_limit = 0.30                             # Linear velocity limit
omega_limit = 3.8                          # Angular velocity limit

# Type of control
# control_type = "setpoint"                  # Options: "setpoint", "traj_tracking"
trajectory = "infinity"                    # Type of trajectory. Options: circular, infinity

# For setpoint control:
Q_sp = np.diag([15, 15, 0.005]) #np.diag([15, 15, 0.005])            # State cost matrix DOORWAY
# Q_sp = np.diag([100, 100, 11]) #np.diag([15, 15, 0.005])            # State cost matrix INTERSECTION
R_sp = np.array([3, 1.5])                  # Controls cost matrix

# For trajectory tracking control:
Q_tr = np.diag([200, 200, 0.005])          # State cost matrix
R_tr = np.array([0.1, 0.001])              # Controls cost matrix

# Obstacles
agent_radius = 0.1                               # Robot radius (for obstacle avoidance)


# ------------------------------------------------------------------------------
COST_MATRICES = {
    "setpoint": {
        "Q": Q_sp,
        "R": R_sp
    },
    "traj_tracking": {
        "Q": Q_tr,
        "R": R_tr,
        "trajectory_params": {
            "circular": {
                "A": 0.8,
                "w": 0.3
            },
            "infinity": {
                "A": 1.0,
                "w": 0.3
            }
        }
    }
}

# ------------------------------------------------------------------------------
# Liveness parameters.
liveliness = False
liveness_threshold = 0.3


