import torch
import config
import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.nn.functional.hardtanh(grad_output)

# class StraightThroughEstimator(torch.nn.Module):
#     def __init__(self):
#         super(StraightThroughEstimator, self).__init__()

#     def forward(self, x):
#         x = STEFunction.apply(x)
#         return x

def angle_between_vectors(v1, v2):
    dot_product = np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2) + EPSILON)
    # print("\tNumpy vel mags:", LA.norm(v1), LA.norm(v2))
    # print("\tNumpy dot products:", np.dot(v1, v2), dot_product)
    return np.arccos(np.clip(dot_product, -1, 1))

# dx and dy are opp_pos - ego_pos
def calculate_is_not_live_torch(dx, dy, ego_theta, ego_v, opp_theta, opp_v):
    ego_vel = torch.tensor([torch.cos(ego_theta), torch.sin(ego_theta)]).to(config.device) * ego_v
    opp_vel = torch.tensor([torch.cos(opp_theta), torch.sin(opp_theta)]).to(config.device) * opp_v

    # Calculate liveness
    vel_diff = opp_vel - ego_vel
    pos_diff = torch.tensor([dx, dy]).to(config.device)
    pos_diff_mag = torch.linalg.vector_norm(pos_diff)
    vel_diff_mag = torch.linalg.vector_norm(vel_diff)
    dot_product = torch.linalg.vecdot(pos_diff, vel_diff) / (pos_diff_mag * vel_diff_mag + EPSILON)
    angle_between = torch.arccos(torch.clamp(dot_product, min=-1.0, max=1.0))
    l = torch.pi - angle_between

    # b(x) = ((x_0 - )) + config.liveness_threshold - torch.pi

    # Check intersection
    ego_vel_uvec = ego_vel / ego_v
    opp_vel_uvec = opp_vel / opp_v
    det = opp_vel_uvec[0] * ego_vel_uvec[1] - opp_vel_uvec[1] * ego_vel_uvec[0]
    u = (dy * opp_vel_uvec[0] - dx * opp_vel_uvec[1]) * det
    v = (dy * ego_vel_uvec[0] - dx * ego_vel_uvec[1]) * det

    # print("Torch pos diff:", pos_diff, vel_diff)
    # print("\tTorch vel magnitudes:", ego_v, opp_v)
    # print("\tTorch pure dot and norm dot:", torch.linalg.vecdot(pos_diff, vel_diff), dot_product)
    # print("\tTorch angle between:", angle_between)
    # print("\tLIVELINESS_TORCH:", l, u, v)

    # is_not_live = l < config.liveness_threshold and u > 0 and v > 0
    # l < config.liveness_threshold
    # config.liveness_threshold > l
    # config.liveness_threshold - l > 0

    is_not_live = STEFunction.apply(config.liveness_threshold - l) * STEFunction.apply(u) * STEFunction.apply(v)
    return is_not_live


def calculate_liveliness(ego_pos, opp_pos, ego_vel, opp_vel):
    vel_diff = opp_vel - ego_vel
    pos_diff = opp_pos - ego_pos
    # print("Numpy pos diff:", pos_diff, vel_diff)
    # print("\tAngle between:", angle_between_vectors(pos_diff, vel_diff))
    l = np.pi - angle_between_vectors(pos_diff, vel_diff)
    ttc = LA.norm(pos_diff) / LA.norm(vel_diff) # Time-to-collision
    # Liveness should only matter if you're both moving towards the collision point.
    return l, ttc, pos_diff, vel_diff


def check_intersection(ego_pos, opp_pos, ego_vel, opp_vel):
    ego_vel_uvec = ego_vel / np.linalg.norm(ego_vel)
    opp_vel_uvec = opp_vel / np.linalg.norm(opp_vel)
    dx = opp_pos[0] - ego_pos[0]
    dy = opp_pos[1] - ego_pos[1]
    det = opp_vel_uvec[0] * ego_vel_uvec[1] - opp_vel_uvec[1] * ego_vel_uvec[0]
    u = (dy * opp_vel_uvec[0] - dx * opp_vel_uvec[1]) * det
    v = (dy * ego_vel_uvec[0] - dx * ego_vel_uvec[1]) * det

    return u > 0 and v > 0


def calculate_all_metrics(ego_state, opp_state, liveness_thresh):
    ego_vel_vec = np.array([np.cos(ego_state[2]), np.sin(ego_state[2])]) * ego_state[3]
    opp_vel_vec = np.array([np.cos(opp_state[2]), np.sin(opp_state[2])]) * opp_state[3]
    l, ttc, pos_diff, vel_diff = calculate_liveliness(ego_state[:2], opp_state[:2], ego_vel_vec, opp_vel_vec)

    intersecting = check_intersection(ego_state[:2], opp_state[:2], ego_vel_vec, opp_vel_vec)

    is_live = False
    if config.consider_intersects:
        if l > liveness_thresh or not intersecting:
            is_live = True
    else:
        if l > liveness_thresh:
            is_live = True

    return l, ttc, pos_diff, vel_diff, intersecting, is_live


def get_x_is_d_goal_input(inputs, goal):
    x = goal[0] - inputs[0]
    y = goal[1] - inputs[1]
    theta = inputs[2]
    v = inputs[3]
    opp_x = inputs[4] - inputs[0] # opp_x - ego_x (ego frame)
    opp_y = inputs[5] - inputs[1] # opp_y - ego_y (ego frame)
    opp_theta = inputs[6]
    opp_v = inputs[7]
    inputs = np.array([x, y, theta, v, opp_x, opp_y, opp_theta, opp_v])
    return inputs

