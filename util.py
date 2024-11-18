import numpy as np
import numpy.linalg as LA

EPSILON = 0.001

def angle_between_vectors(v1, v2):
    return np.arccos(np.dot(v1, v2) / (LA.norm(v1) * LA.norm(v2) + EPSILON))

def intersect_lines(A, d1, C, d2):
    # Convert points and directions to NumPy arrays
    A, d1, C, d2 = map(np.array, (A, d1, C, d2))

    # Construct the coefficient matrix and the right-hand side vector
    matrix = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
    rhs = C - A

    # Check if the matrix is invertible
    if np.linalg.det(matrix) == 0:
        return None  # Lines are parallel or coincident

    # Solve for t and u
    t, u = np.linalg.solve(matrix, rhs)

    # Compute the intersection point using t
    intersection = A + t * d1
    return tuple(intersection)

def calculate_liveliness(ego_state, opp_state):
    v_ego = np.round(ego_state[3], 2)
    v_opp = np.round(opp_state[3], 2)
    ego_vel = np.array([v_ego * np.cos(ego_state[2]), v_ego * np.sin(ego_state[2])])
    opp_vel = np.array([v_opp * np.cos(opp_state[2]), v_opp * np.sin(opp_state[2])])
    vel_diff = ego_vel - opp_vel
    pos_diff = ego_state[:2] - opp_state[:2]
    l = np.arccos(-(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
    ttc = LA.norm(pos_diff) / LA.norm(vel_diff) # Time-to-collision
    # Liveness should only matter if you're both moving towards the collision point.
    return l, ttc, pos_diff, vel_diff

def calculate_all_metrics(ego_state, opp_state):
    l, ttc, pos_diff, vel_diff = calculate_liveliness(ego_state, opp_state)
    ego_vel_vec = np.array([np.cos(ego_state[2]), np.sin(ego_state[2])]) * ego_state[3]
    opp_vel_vec = np.array([np.cos(opp_state[2]), np.sin(opp_state[2])]) * opp_state[3]
    intersect_point = intersect_lines(ego_state[:2], ego_vel_vec, opp_state[:2], opp_vel_vec)
    if intersect_point is None:
        intersecting = False
    else:
        egoanglediff = abs(angle_between_vectors(intersect_point - ego_state[:2], ego_vel_vec))
        oppanglediff = abs(angle_between_vectors(intersect_point - opp_state[:2], opp_vel_vec))
        # print(f"Ego angle diff: {egoanglediff}, Opp angle diff: {oppanglediff}")
        intersecting = (egoanglediff <= np.pi/2) and (oppanglediff <= np.pi/2)

    return l, ttc, pos_diff, vel_diff, intersecting


def get_x_is_d_goal_input(inputs, goal):
    x = goal[0] - inputs[0]
    y = goal[1] - inputs[1]
    theta = inputs[2]
    v = inputs[3]
    opp_x = inputs[4] - inputs[0]
    opp_y = inputs[5] - inputs[1]
    opp_theta = inputs[6]
    opp_v = inputs[7]
    inputs = np.array([x, y, theta, v, opp_x, opp_y, opp_theta, opp_v])
    return inputs

