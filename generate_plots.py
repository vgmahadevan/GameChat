import config
import numpy as np
from metrics import check_when_reached_goal, get_straight_line_desired_path
from data_logger import DataLogger
from run_experiments import get_scenario
from util import get_ray_intersection_point
import matplotlib.pyplot as plt

def get_liveness_cbf(ego_state, opp_state, is_faster):
    center_intersection = get_ray_intersection_point(ego_state[:2], ego_state[2], opp_state[:2], opp_state[2])
    vec_to_opp = opp_state[:2] - ego_state[:2]
    unit_vec_to_opp = vec_to_opp / np.linalg.norm(vec_to_opp)
    initial_closest_to_opp = ego_state[:2] + unit_vec_to_opp * (config.agent_radius)
    opp_closest_to_initial = opp_state[:2] - unit_vec_to_opp * (config.agent_radius)
    intersection = get_ray_intersection_point(initial_closest_to_opp, ego_state[2], opp_closest_to_initial, opp_state[2])
    if center_intersection is None or intersection is None or ego_state[3] == 0 or opp_state[3] == 0:
        return None

    d0 = np.linalg.norm(initial_closest_to_opp - intersection)
    d1 = np.linalg.norm(opp_closest_to_initial - intersection)

    if is_faster: # Ego agent is faster
        barrier = d1 / opp_state[3] - d0 / ego_state[3]
    else: # Ego agent is slower
        barrier = d0 / ego_state[3] - d1 / opp_state[3]
    return barrier

global figcount
figcount = 1

def gen_figure(ys, title, labels, ylabel, filesuffix, ylims=None, add_dotted=None):
    global figcount
    plt.figure(figcount)
    figcount += 1
    plt.title(title)
    for y, label in zip(ys, labels):
        plt.plot(y, label = label)
    
    if add_dotted is not None:
        val, label = add_dotted
        plt.plot([val if i is not None else None for i in ys[0]], label=label, linestyle='--', color='red')

    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    plt.legend()

    if ylims is not None:
        plt.ylim(ylims)

    plt.savefig(f'experiment_results/histories/{RUN_AGENT}_{SCENARIO}_{filesuffix}.png')
    print("Saved plot to", f"experiment_results/histories/{RUN_AGENT}_{SCENARIO}_{filesuffix}.png")

def gen_traj_plot(desireds, trajs, labels, title, filesuffix, plot_skip = 3):
    global figcount
    plt.figure(figcount)
    figcount += 1
    plt.title(title)
    for desired, traj, label in zip(desireds, trajs, labels):
        plt.plot(desired[:, 0], desired[:, 1], linestyle='--', label = label)
        plt.scatter(traj[::plot_skip, 0], traj[::plot_skip, 1], label = label)
    

    plt.ylabel('Y (m)')
    plt.xlabel('X (m)')
    plt.legend()

    plt.savefig(f'experiment_results/desired_paths/{RUN_AGENT}_{SCENARIO}_{filesuffix}.png')
    print("Saved plot to", f"experiment_results/desired_paths/{RUN_AGENT}_{SCENARIO}_{filesuffix}.png")


SCENARIO = 'Doorway'
# SCENARIO = 'Intersection'

# RUN_AGENT = 'MPC'
RUN_AGENT = 'LiveNet'

filename = f'experiment_results/histories/{RUN_AGENT}_{SCENARIO}.json'
logger = DataLogger.load_file(filename)

static_obs = np.array(logger.data['obstacles'])

obs_dist_vals_0 = []
obs_dist_vals_1 = []
opp_dist_vals = []

liveness_cbf_vals = []
vels_0 = []
vels_1 = []

logger0 = DataLogger.load_file(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_0.json")
desired0 = np.array([iteration['states'][0] for iteration in logger0.data['iterations']])
logger1 = DataLogger.load_file(f"experiment_results/desired_paths/{SCENARIO}_{RUN_AGENT}_1.json")
desired1 = np.array([iteration['states'][1] for iteration in logger1.data['iterations']])

traj0 = np.array([iteration['states'][0] for iteration in logger.data['iterations']])
traj1 = np.array([iteration['states'][1] for iteration in logger.data['iterations']])
first_reached_goal = check_when_reached_goal(traj0, logger.data['iterations'][0]['goals'][0][:2])
second_reached_goal = check_when_reached_goal(traj1, logger.data['iterations'][0]['goals'][1][:2])

for iteration in logger.data['iterations']:
    ego_state = np.array(iteration['states'][0])
    opp_state = np.array(iteration['states'][1])

    vels_0.append(ego_state[3])
    vels_1.append(opp_state[3])

    obs_dist_vals_0.append(np.min(np.linalg.norm(static_obs[:, :2] - ego_state[:2], axis=1) ** 2.0 - (config.agent_radius + static_obs[:, 2]) ** 2.0))
    obs_dist_vals_1.append(np.min(np.linalg.norm(static_obs[:, :2] - opp_state[:2], axis=1) ** 2.0 - (config.agent_radius + static_obs[:, 2]) ** 2.0))
    opp_dist_vals.append(np.linalg.norm(ego_state[:2] - opp_state[:2]) ** 2 - (config.agent_radius * 2.0) ** 2.0)

    liveness_cbf_vals.append(get_liveness_cbf(ego_state, opp_state, first_reached_goal < second_reached_goal))

first_name = "Faster" if first_reached_goal < second_reached_goal else "Slower"
second_name = "Slower" if first_reached_goal < second_reached_goal else "Faster"
gen_figure([vels_0, vels_1], "Agent Velocity", [f'{first_name} Agent', f'{second_name} Agent'], "Agent Velocity (m/s)", "velocities")
gen_figure([obs_dist_vals_0, obs_dist_vals_1, opp_dist_vals], "Distance CBF Violation", [f'{first_name} Agent Static Obstacle Distance', f'{second_name} Agent Static Obstacle Distance', 'Inter-Agent Distance'],  "Distance (m)", "distance_cbf", add_dotted=(0, 'CBF Boundary'), ylims=[-0.2, np.max([obs_dist_vals_0, obs_dist_vals_1, opp_dist_vals])])
gen_figure([liveness_cbf_vals], "Liveness CBF Violation", [f'Agent Liveness'],  "Liveness (s)", "liveness_cbf", add_dotted=(0, 'CBF Boundary'), ylims=[-0.2, 20])
gen_traj_plot([desired0, desired1], [traj0, traj1], [f"{first_name} Agent", f"{second_name} Agent"], "Desired vs. Taken Trajectories", "desired")
