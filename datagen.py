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
from data_logger import DataLogger, BlankLogger
from environment import Environment
from simulation import run_simulation

folder_to_save_to = 'all_data_with_offsets/'

# offset = [0, 1, 3, 5, 7, -1, -3, -5, -7]
offset = [0]
zero_faster = [True, False]
for z in zero_faster:
    for o in offset:
        config.agent_zero_offset = o
        config.mpc_p0_faster = z
        config.save_data_path = f'test_doorway_train_data_with_liveness_{0 if z else 1}_faster_off{o}.json'

        scenario = DoorwayScenario()

        # Matplotlib plotting handler
        # plotter = Plotter()
        plotter = None
        if config.save_data_path is None:
            logger = BlankLogger()
        else:
            logger = DataLogger(os.path.join(folder_to_save_to, config.save_data_path))

        # Add all initial and goal positions of the agents here (Format: [x, y, theta])
        goals = scenario.goals.copy()
        logger.set_obstacles(scenario.obstacles.copy())
        env = Environment(scenario.initial.copy(), scenario.goals.copy())
        controllers = []

        # Setup agents
        controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
        controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))

        run_simulation(scenario, env, controllers, logger, plotter)
        plt.close()
