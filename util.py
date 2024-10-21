import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

def calculate_liveliness(ego_state, opp_state):
    ego_vel = np.array([ego_state[3] * np.cos(ego_state[2]), ego_state[3] * np.sin(ego_state[2])])
    opp_vel = np.array([opp_state[3] * np.cos(opp_state[2]), opp_state[3] * np.sin(opp_state[2])])
    vel_diff = ego_vel - opp_vel
    pos_diff = ego_state[:2] - opp_state[:2]
    l = np.arccos(abs(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
    return l, pos_diff, vel_diff
