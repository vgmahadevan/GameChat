"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
from plotter import Plotter
from data_logger import DataLogger
from environment import Environment
from blank_controller import BlankController
from simulation import run_simulation
from run_experiments import get_mpc_live_controllers, get_scenario

NUM_AGENTS = 2

SCENARIO = "Doorway"
# SCENARIO = "Intersection"

scenario = get_scenario(SCENARIO)

config.obstacle_avoidance = True
config.mpc_use_opp_cbf = False
config.liveliness = False
config.liveliness_gamma = 1.0
config.liveness_threshold = 1.0
config.obs_gamma = 1.0
config.mpc_static_obs_non_cbf_constraint = True

for agent_idx in range(1):
    config.mpc_p0_faster = agent_idx == 0
    plotter = Plotter()
    config.ani_save_name = f"TEST_{agent_idx}.mp4"
    logger = DataLogger(f"experiment_results/desired_paths/{SCENARIO}_agent_{agent_idx}_test.json")

    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    controllers = [BlankController() for _ in range(NUM_AGENTS)]
    controllers[agent_idx] = get_mpc_live_controllers(scenario, True)[agent_idx]

    run_simulation(scenario, env, controllers, logger, plotter)
