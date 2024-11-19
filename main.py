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
scenario = DoorwayScenario(initial_x=-1.0, initial_y=0.5, goal_x=2.0, goal_y=.15)
config.opp_gamma = 0.1
config.obs_gamma = 0.1
config.liveliness_gamma = 0.2
config.liveness_threshold = 1.0
# scenario = NoObstacleDoorwayScenario(rotation=np.pi/2)
# scenario = NoObstacleDoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
# plotter = Plotter()
plotter = None
logger = BlankLogger()

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())
env = Environment(scenario.initial.copy(), scenario.goals.copy())
controllers = []

# Setup agent 0
# controllers.append(BlankController())
# controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model2_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_obs_l_f_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))

# Setup agent 1
# controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))
controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_liveness_1_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_newb_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_test2_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_smg_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_base_w_lims_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_base_w_lims_opp_pen_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_smg_w_lims_opp_pen_dgoal_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_base_w_lims_opp_pen_dgoal_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_smg_w_lims_opp_pen_dgoal_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model2_smg_w_lims_opp_pen_dgoal_obs_l_s_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model20_smg_w_lims_opp_pen_dgoal_obs_l_saf_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_20_base_w_lims_opp_pen_dgoal_obs_l_all_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_30_smg_w_lims_opp_pen_dgoal_obs_l_all_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_30_smgbin_w_lims_opp_pen_dgoal_obs_l_all_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_25_smgbin_w_lims_opp_pen_dgoal_fixo_obs_l_all_more_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))

run_simulation(scenario, env, controllers, logger, plotter)
