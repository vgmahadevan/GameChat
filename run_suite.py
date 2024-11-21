"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import os
import matplotlib.pyplot as plt
import config
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import BlankLogger
from model_controller import ModelController
from environment import Environment
from simulation import run_simulation

VIZ = False

# start x, start y, goal x, goal y, opp gamma, obs gamma, liveness gamma
scenario_configs = [
    (-1.0, 0.5, 2.0, 0.15), # 0
    (-1.0, 0.5, 2.0, 0.30), # 1
    (-1.0, 0.5, 2.0, 0.45), # 2
    (-1.0, 0.4, 2.0, 0.15), # 3
    (-1.0, 0.4, 2.0, 0.35), # 4
    (-1.0, 0.3, 2.0, 0.15), # 5
    (-1.0, 0.3, 2.0, 0.25), # 6

    (-0.5, 0.5, 2.0, 0.15), # 7
    (-0.5, 0.5, 2.0, 0.30), # 8
    (-0.5, 0.5, 2.0, 0.45), # 9
    (-0.5, 0.4, 2.0, 0.15), # 10
    (-0.5, 0.4, 2.0, 0.35), # 11
    (-0.5, 0.3, 2.0, 0.15), # 12
    (-0.5, 0.3, 2.0, 0.25), # 13
]

for scenario_config in scenario_configs:
    print(f"Running scenario {scenario_config}")
    scenario = DoorwayScenario(initial_x=scenario_config[0], initial_y=scenario_config[1], goal_x=scenario_config[2], goal_y=scenario_config[3])

    logger = BlankLogger()
    plotter = Plotter()
    # plotter = None

    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    goals = scenario.goals.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    env = Environment(scenario.initial.copy(), scenario.goals.copy())
    controllers = []

    # Setup agents
    controllers.append(ModelController("weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
    controllers.append(ModelController("weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
    # controllers.append(MPC(agent_idx=0, goal=goals[0,:], opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
    # controllers.append(MPC(agent_idx=1, goal=goals[1,:], opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))

    run_simulation(scenario, env, controllers, logger, plotter)
    plt.close()
