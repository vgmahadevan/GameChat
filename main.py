"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
from mpc_cbf import MPC
from scenarios import DoorwayScenario, NoObstacleDoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger, BlankLogger
from environment import Environment
from model_controller import ModelController
from blank_controller import BlankController
from simulation import run_simulation

# Scenarios: "doorway" or "intersection"
scenario = DoorwayScenario()
# scenario = NoObstacleDoorwayScenario(rotation=np.pi/2)
# scenario = NoObstacleDoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()
logger = BlankLogger()

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())
env = Environment(scenario.initial.copy(), scenario.goals.copy())
controllers = []

# Setup agent 0
# controllers.append(BlankController())
# controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
controllers.append(MPC(agent_idx=0, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model2_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_obs_l_f_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))

# Setup agent 1
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))
# controllers.append(MPC(agent_idx=1, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_1_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
controllers.append(ModelController("weights/model_test_obs_l_f_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))

run_simulation(scenario, env, controllers, logger, plotter)
