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
from data_logger import BlankLogger
from environment import Environment
from blank_controller import BlankController
from model_controller import ModelController
from simulation import run_simulation
from metrics import gather_all_metric_data

# config.opp_gamma = 0.65
# config.runtime = 14.0
# config.liveliness_threshold = 0.5
# zeta = 2.0
# scenario = IntersectionScenario(start=0.8, goal=1.2)
# scenario = IntersectionScenario(start=1.0, goal=1.0)
config.liveliness_gamma = 0.1
scenario_params = (1.0, 1.0)
scenario = IntersectionScenario(start=scenario_params[0], goal=scenario_params[1])

# scenarios = [
#     (0.8, 0.8)
#     (1.0, 1.0)
#     (1.2, 1.2)
#     (0.8, 1.0)
#     (0.8, 1.2)
#     (1.0, 0.8)
#     (1.0, 1.2)
#     (1.2, 0.8)
#     (1.2, 1.0)
# ]
# params = [
#     (0.65, 0.5, 2.0)
#     (0.5, 0.5, 2.0)
#     (0.65, 0.5, 2.0)
#     (0.65, 0.5, 2.0)
#     (0.65, 0.5, 2.0)
#     (0.5, 0.5, 2.0)
#     (0.5, 0.5, 2.0)
#     (0.65, 0.5, 2.0)
#     (0.65, 0.5, 2.0)
# ]

# scenario_params = (-1.0, 0.5, 2.0, 0.15)
# scenario = DoorwayScenario(initial_x=scenario_params[0], initial_y=scenario_params[1], goal_x=scenario_params[2], goal_y=scenario_params[3])

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
# controllers.append(MPC(agent_idx=0, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[0,:], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/livetest8_intersection_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Doorway livenet
# controllers.append(ModelController("weights/model_30_norm_doorsuite2_lf_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Doorway livenet

# controllers.append(ModelController("weights/model_30_norm_doorsuite2_lfnew_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Doorway livenet

controllers.append(ModelController("weights/model_40_norm_intersuite2_lfnew_nso_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite3_lfnew_nso_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite2_lfnew_nso_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite2_lfnew_so_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite3_lfnew_so_0_1_bn_definition.json", goals[0], static_obs=scenario.obstacles.copy())) # Intersection livenet


# Setup agent 1
# controllers.append(BlankController())
# controllers.append(MPC(agent_idx=1, opp_gamma=config.opp_gamma, obs_gamma=config.obs_gamma, live_gamma=config.liveliness_gamma, liveness_thresh=config.liveness_threshold, goal=goals[1,:], static_obs=scenario.obstacles.copy()))

# controllers.append(ModelController("weights/model_base_input_obs_wc_nolim_saf_intersuite_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection base barriernet
# controllers.append(ModelController("weights/model_base_single_input_obs_wc_nolim_saf_suite_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))
# controllers.append(ModelController("weights/model_base_single_input_obs_wc_nolim_linp_f_fullsuite_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy()))

# controllers.append(ModelController("weights/livetest8_intersection_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Doorway livenet
# controllers.append(ModelController("weights/model_30_norm_doorsuite2_lf_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Doorway livenet

# controllers.append(ModelController("weights/model_30_norm_doorsuite2_lfnew_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Doorway livenet

controllers.append(ModelController("weights/model_40_norm_intersuite2_lfnew_nso_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite3_lfnew_nso_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite2_lfnew_nso_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite2_lfnew_so_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection livenet
# controllers.append(ModelController("weights/model_30_norm_intersuite3_lfnew_so_0_1_bn_definition.json", goals[1], static_obs=scenario.obstacles.copy())) # Intersection livenet

x_cum, u_cum = run_simulation(scenario, env, controllers, logger, plotter)

metric_data = gather_all_metric_data(scenario, x_cum[0], x_cum[1], scenario.goals)
print((config.opp_gamma, config.obs_gamma, config.liveliness_gamma, config.liveness_threshold), metric_data)
