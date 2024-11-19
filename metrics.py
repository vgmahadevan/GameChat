import config
import numpy as np

def dist_between_line_and_point(target_point, line_point, line_heading):
    cos_heading = np.cos(line_heading)
    sin_heading = np.sin(line_heading)

    dx = target_point[0] - line_point[0]
    dy = target_point[1] - line_point[1]

    distance = np.abs(cos_heading * dy - sin_heading * dx)
    
    return distance

def get_desired_path(start, goal, discretization=0.01):
    number_of_points = int(np.linalg.norm(start - goal) / discretization) + 1
    xs = np.linspace(start[0] , goal[0], number_of_points)
    ys = np.linspace(start[1], goal[1], number_of_points)
    return np.array([xs, ys]).T


def calculate_path_deviation(traj, desired_path):
    total_deviation = 0
    for state in traj:
        path_deviation_array = np.linalg.norm(state[:2] - desired_path, axis=1)
        closest_idx = np.argmin(path_deviation_array)
        path_deviation_array[closest_idx] = np.inf
        second_closest_idx = np.argmin(path_deviation_array)

        path1 = desired_path[closest_idx]
        path2 = desired_path[second_closest_idx]
        line_heading = np.arctan2(path2[1] - path1[1], path2[0] - path1[0])
        deviation = dist_between_line_and_point(state[:2], desired_path[closest_idx], line_heading)

        total_deviation += deviation
    return total_deviation


def check_when_reached_goal(traj, goal):
    for iteration, state in enumerate(traj):
        # If we've reached the goal and come to a stop, stop counting.
        if np.linalg.norm(state[:2] - goal) < 0.05 and np.abs(state[3]) < 0.05:
            return iteration

    return None


def calculate_avg_delta_vel(traj):
    total_delta_v = 0.0
    prev_v = traj[0][3]
    count = 0
    for state in traj:
        total_delta_v += abs(state[3] - prev_v)
        prev_v = state[3]
        count += 1
    return total_delta_v / count


def check_for_traj_collisions(traj1, traj2):
    for state1, state2 in zip(traj1, traj2):
        dist = np.linalg.norm(state1[:2] - state2[:2])
        if dist < config.agent_radius * 2 + config.safety_dist:
            return True

    return False


def check_for_obs_collisions(traj, obstacles):
    for state in traj:
        for obs_x, obs_y, obs_r in obstacles:
            dist = np.linalg.norm(state[:2] - np.array([obs_x, obs_y]))
            if dist < config.agent_radius + obs_r + config.safety_dist:
                return True

    return False
