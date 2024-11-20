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
togen = [
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




    # (0.0, 0.3, 3.0, 0.15), # 30

]
scenario_params = togen[0]
# scenario_params = (0.0, 0.3, 3.0, 0.15)

# Second to last one is whether or not to turn on the obstacle CBF.
# Last one is runtime.
# Best params: [
#   0: (0.5, 0.3, 0.3, 1.0, True, 18.0?)
#   1: (0.8, 0.4, 0.4, 1.0, True, 17.0)
#   2: ???
#   3: (0.7, 0.6, 0.1, 1.3, True, 18.0)
#   4: (0.7, 0.6, 0.2, 1.3, True, 18.0)
#   5: (0.8, 0.6, 0.2, 1.4, True, 16.0)
#   6: (0.8, 0.6, 0.2, 1.0, False, 16.0)
#   7: (0.9, 0.3, 0.2, 0.7, True, 15.0)
#   8: (0.9, 0.2, 0.2, 0.8, False, 16.0)
#   9: (0.9, 0.3, 0.2, 0.8, False, 16.0)
#   10: (0.9, 0.3, 0.2, 0.9, False, 14.0)
#   11: (0.9, 0.3, 0.2, 0.9, False, 15.0)
#   12: (0.9, 0.3, 0.2, 1.1, False, 14.0)
#   13: (0.9, 0.3, 0.2, 1.1, False, 15.0)





#   30: (0.6, 0.2, 0.5, 1.0, True, 18.0)



# ]

scenario = DoorwayScenario(initial_x=scenario_params[0], initial_y=scenario_params[1], goal_x=scenario_params[2], goal_y=scenario_params[3])
# config.opp_gamma = 0.9
# config.obs_gamma = 0.3
# config.liveliness_gamma = 0.2
# config.liveness_threshold = 1.1
delay_start = 0.0

config.opp_gamma = 0.6
config.obs_gamma = 0.2
config.liveliness_gamma = 0.5
config.liveness_threshold = 1.0


# scenario = NoObstacleDoorwayScenario(rotation=np.pi/2)
# scenario = NoObstacleDoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()
# plotter = None
logger = BlankLogger()

# Add all initial and goal positions of the agents here (Format: [x, y, theta])
goals = scenario.goals.copy()
logger.set_obstacles(scenario.obstacles.copy())
env = Environment(scenario.initial.copy(), scenario.goals.copy())
controllers = []

# Setup agent 0
# controllers.append(BlankController())
# controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=max(config.agent_zero_offset, 0.0)))
controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy(), delay_start=delay_start))
# controllers.append(ModelController("weights/model_liveness_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_0_bn_definition.json", static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_l_saf_g_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_o_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model2_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_l_saf_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_obs_l_f_0_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model4_25_smgbin_l_w_lims_opp_pen_dgoal_fixo_obs_l_suite_multi_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy()))

# Setup agent 1
# controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy(), delay_start=max(-config.agent_zero_offset, 0.0)))
# controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy()))
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
# controllers.append(ModelController("weights/model_25_smgbin_suite_w_lims_opp_pen_dgoal_fixo_obs_l_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
controllers.append(ModelController("weights/model4_25_smgbin_l_w_lims_opp_pen_dgoal_fixo_obs_l_suite_multi_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))

x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)

from metrics import gather_all_metric_data
metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals)
print((config.opp_gamma, config.obs_gamma, config.liveliness_gamma, config.liveness_threshold), metric_data)
