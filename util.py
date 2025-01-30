import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

def calculate_liveliness(ego_state, opp_state):
    ego_vel = np.array([ego_state[3] * np.cos(ego_state[2]), ego_state[3] * np.sin(ego_state[2])])
    opp_vel = np.array([opp_state[3] * np.cos(opp_state[2]), opp_state[3] * np.sin(opp_state[2])])
    vel_diff = ego_vel - opp_vel
    pos_diff = ego_state[:2] - opp_state[:2]
    l = np.arccos(abs(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
    ttc = LA.norm(pos_diff) / LA.norm(vel_diff) # Time-to-collision
    col_pt = get_intersection_point(ego_state, opp_state)
    return l, ttc, pos_diff, vel_diff, col_pt


def check_ray_intersection(ego_state, opp_state):
    x_ego, y_ego, theta_ego, _ = ego_state
    x_opp, y_opp, theta_opp, _ = opp_state

    # Direction vectors for each ray
    xvel_ego = np.cos(theta_ego)
    yvel_ego = np.sin(theta_ego)
    xvel_opp = np.cos(theta_opp)
    yvel_opp = np.sin(theta_opp)
    
    det = xvel_ego * yvel_opp - yvel_ego * xvel_opp 

    # If the determinant is zero, the rays are parallel or collinear
    if abs(det) < 1e-6:
        return False, None 

    t = ((x_opp - x_ego) * yvel_opp - (y_opp - y_ego) * xvel_opp) / det
    s = ((x_opp - x_ego) * yvel_ego - (y_opp - y_ego) * xvel_ego) / det

    # If both t >= 0 and s >= 0, the rays intersect

    return (t >= 0 and s >= 0, t)

def get_intersection_point(ego_state, opp_state):
    intersect, t = check_ray_intersection(ego_state, opp_state)

    if not intersect:
        return None 

    x_ego, y_ego, theta_ego, _ = ego_state
    intersection_x = x_ego + t * np.cos(theta_ego)
    intersection_y = y_ego + t * np.sin(theta_ego)

    intersection_pt = [intersection_x, intersection_y]
    return intersection_pt
