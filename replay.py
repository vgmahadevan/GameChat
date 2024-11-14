"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import numpy as np
import config
from scenarios import DoorwayScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from util import calculate_all_metrics

scenario = DoorwayScenario()

# Matplotlib plotting handler
plotter = Plotter()
# logger = DataLogger.load_file('doorway_train_data_with_liveness_0_faster.json')
# logger = DataLogger.load_file('all_data_with_offsets/test_doorway_train_data_with_liveness_0_faster_off0.json')
logger = DataLogger.load_file('all_data_with_offsets/doorway_train_data_with_liveness_0_faster_off0.json')
output_logger = BlankLogger()

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())

x_cum = [[], []]
u_cum = [[], []]
metrics = []
for sim_iteration, iteration in enumerate(logger.data['iterations']):
    print(f"\nIteration: {sim_iteration}")
    for agent_idx, state in enumerate(iteration['states']):
        x_cum[agent_idx].append(np.array(state))

    for agent_idx, controls in enumerate(iteration['controls']):
        u_cum[agent_idx].append(np.array(controls))

    metrics.append(calculate_all_metrics(x_cum[0][-1], x_cum[1][-1]))

    # Plots
    if sim_iteration % config.plot_rate == 0 and config.plot_live and plotter is not None:
        plotter.plot_live(scenario, x_cum, u_cum, metrics)

# Discard the first element of both x1 and x2
x_cum = np.array(x_cum)
u_cum = np.array(u_cum)
if config.plot_end and plotter is not None:
    plotter.plot(scenario, x_cum, u_cum, metrics)

