import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

def calculate_liveliness(ego_state, opp_state):
    v_ego = np.round(ego_state[3], 2)
    v_opp = np.round(opp_state[3], 2)
    ego_vel = np.array([v_ego * np.cos(ego_state[2]), v_ego * np.sin(ego_state[2])])
    opp_vel = np.array([v_opp * np.cos(opp_state[2]), v_opp * np.sin(opp_state[2])])
    vel_diff = ego_vel - opp_vel
    pos_diff = ego_state[:2] - opp_state[:2]
    l = np.arccos(-(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
    ttc = LA.norm(pos_diff) / LA.norm(vel_diff) # Time-to-collision
    return l, ttc, pos_diff, vel_diff
