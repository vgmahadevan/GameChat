import do_mpc
from casadi import *
import config
from config import DynamicsModel, Role
import numpy as np
from util import calculate_liveliness, EPSILON

class MPC:
    """MPC-CBF Optimization problem:

    min Σ_{k=0}{N-1} 1/2*x'_k^T*Q*x'_k + 1/2*u_k^T*R*u_k   over u
    s.t.
        x_{k+1} = x_k + B*u_k*T_s
        x_min <= x_k <= x_max
        u_min <= u_k <= u_max
        x_0 = x(0)
        Δh(x_k, u_k) >= -γ*h(x_k)

    where x'_k = x_{des_k} - x_k
    """
    def __init__(self, agent_idx, goal, static_obs = [], opp_state = None):
        self.agent_idx = agent_idx
        self.goal = goal
        self.static_obs = static_obs
        self.opp_state = opp_state
        self.Q = config.COST_MATRICES[config.dynamics]['Q']
        self.R = config.COST_MATRICES[config.dynamics]['R']
        self.u_ori = []
        self.liveliness = []
        self.last_liveliness_iteration = -10
        self.role = None

    def initialize_controller(self, env):
        self.model = env.define_model()
        self.env = env

    """Defines the objective function wrt the state cost depending on the type of control."""
    def get_cost_expression(self, model):
        # Define state error
        X = model.x['x'] - self.goal
        cost_expression = transpose(X)@self.Q@X

        return cost_expression

    """Configures the mpc controller."""
    def define_mpc(self):
        mpc = do_mpc.controller.MPC(self.model)

        # Set parameters
        setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': config.T_horizon,
                     't_step': config.Ts,
                     'state_discretization': 'discrete',
                     'store_full_solution': True,
                     'nlpsol_opts': {'ipopt.print_level':0, 'print_time':0},
                    #  'nlpsol_opts': {'ipopt.print_level':6, 'print_time':0},
                     # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
                     }
        if self.agent_idx == 1:
            setup_mpc['nlpsol_opts'] = {'ipopt.print_level':0, 'print_time':0}
        mpc.set_param(**setup_mpc)

        # Configure objective function
        mterm = self.get_cost_expression(self.model)
        lterm = mterm

        mpc.set_objective(mterm=mterm, lterm=lterm)
        mpc.set_rterm(u=self.R)         # Input penalty (R diagonal matrix in objective fun)

        # State and input bounds
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            max_u = np.array([config.v_limit, config.omega_limit])
            mpc.bounds['lower', '_u', 'u'] = -max_u
            mpc.bounds['upper', '_u', 'u'] = max_u
        else:
            max_u = np.array([config.accel_limit, config.omega_limit])
            mpc.bounds['lower', '_u', 'u'] = -max_u
            mpc.bounds['upper', '_u', 'u'] = max_u

            max_x = np.array([float("inf"), float("inf"), float("inf"), config.v_limit])
            mpc.bounds['lower', '_x', 'x'] = -max_x
            mpc.bounds['upper', '_x', 'x'] = max_x


        # MPC-CBF: Add CBF safety constraints
        if config.obstacle_avoidance:
            self.add_cbf_constraints(mpc)

        # if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR and config.liveliness and self.agent_idx == 1:
        #     self.add_liveliness_constraint(mpc)
        if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR and config.liveliness:
            self.add_liveliness_constraint(mpc)

        mpc.setup()
        return mpc

    def add_cbf_constraints(self, mpc):
        cbf_constraints = self.get_cbf_constraints()
        for i, cbc in enumerate(cbf_constraints):
            mpc.set_nl_cons('cbf_constraint'+str(i), cbc, ub=0)

    """Computes the CBF constraints for all obstacles."""
    def get_cbf_constraints(self):
        # Get state vector x_{t+k+1}
        A, B = self.env.get_dynamics(self.model.x['x'])
        x_k1 = self.model.x['x'] + A*config.Ts + B@self.model.u['u']*config.Ts

        # Compute CBF constraints
        cbf_constraints = []

        obs_to_add = self.static_obs.copy()

        if not config.liveliness:
            opponent_obs = (self.opp_state[0], self.opp_state[1], config.agent_radius)
            obs_to_add.append(opponent_obs)

        if self.agent_idx == 1:
            for obs in obs_to_add:
                # delta_h_k + gamma*h_k >= 0
                # h_k1 - h_k + gamma*h_k >= 0
                # -h_k1 + h_k - gamma*h_k <= 0
                # -h_k1 + (1 - gamma)*h_k <= 0
                h_k = self.h_obs(self.model.x['x'], obs)
                h_k1 = self.h_obs(x_k1, obs)
                cbf_constraints.append(-h_k1 + (1-config.obs_gamma)*h_k)
        else:
            # opponent_obs = (self.opp_state[0], self.opp_state[1], config.agent_radius)
            # for obs in self.static_obs + [opponent_obs]:
            for obs in obs_to_add:
                # delta_h_k + gamma*h_k >= 0
                # h_k1 - h_k + gamma*h_k >= 0
                # -h_k1 + h_k - gamma*h_k <= 0
                # -h_k1 + (1 - gamma)*h_k <= 0
                h_k = self.h_obs(self.model.x['x'], obs)
                h_k1 = self.h_obs(x_k1, obs)
                cbf_constraints.append(-h_k1 + (1-config.obs_gamma)*h_k)

        return cbf_constraints
    

    """Computes the Control Barrier Function for an obstacle."""
    def h_obs(self, x, obstacle):
        x_obs, y_obs, r_obs = obstacle
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (config.agent_radius + r_obs + config.safety_dist)**2
        return h
    
    # Assumes that the double-integrator dynamic model is being used
    def add_liveliness_constraint(self, mpc):
        if self.opp_state is None:
            return

        l, ttc, _, _ = calculate_liveliness(self.initial_state.copy(), self.opp_state)
        if l > config.liveness_threshold:
            return
        
        #print(f"Adding constraint, liveliness = {l}")

        # Get state vector x_{t+k+1}
        A, B = self.env.get_dynamics(self.model.x['x'])
        x_k1 = self.model.x['x'] + A*config.Ts + B@self.model.u['u']*config.Ts

        # Compute CBF constraints
        h_k = self.h_v(self.model.x['x'], self.opp_state)
        h_k1 = self.h_v(x_k1, self.opp_state)
        constraint = -h_k1 + (1-config.liveliness_gamma)*h_k
        # -h_k1 + (1 - gamma)*h_k <= 0
        # h_k1 >= h_k - gamma*h_k
        # (h_k1 - h_k) >= -gamma*h_k
        #print("\tConstraint:", constraint)
        mpc.set_nl_cons('liveliness_constraint', constraint, ub=0)

    def h_v(self, x, opp_x):
        self.A_matrix = SX.zeros(2, 2)
        max_zeta = 0.3 / opp_x[3]
        upper_zeta = min(max_zeta, config.zeta)
        self.A_matrix[0, 0] = 1.0
        self.A_matrix[0, 1] = -upper_zeta
        self.A_matrix[1, 0] = -config.zeta
        self.A_matrix[1, 1] = 1.0

        # ego_v - 3 * opp_v >= 0.0 -> ego_v >= 3 * opp_v
        # opp_v - 3 * ego_v >= 0.0 -> ego_v <= 1/3 * opp_v
        # Means that agent 1 will speed up and agent 2 will slow down.
    
        vel_vector = vertcat(x[3], opp_x[3])
        #print(vel_vector)
        h_vec = self.A_matrix @ vel_vector
        h = h_vec[self.agent_idx]
        #print(h)
        # h = mmax(h_vec)
        return h

    """Sets the initial state in all components."""
    def reset_state(self, initial_state, opp_state):
        self.initial_state = initial_state
        self.opp_state = opp_state
        self.mpc = self.define_mpc()
        self.mpc.setup()
        self.mpc.reset_history()
        self.mpc.x0 = self.initial_state
        self.mpc.u0 = np.zeros_like(self.mpc.u0['u'])
        self.mpc.set_initial_guess()

    def make_step(self, x0, mpccbf=False):
        u1 = self.mpc.make_step(x0)
        u1[0] = min(u1[0], config.v_limit)

        # Add liveliness condition here
        ego_state = self.initial_state.copy()
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            ego_state = np.append(ego_state, [u1[0][0]])
        l, ttc, pos_diff, vel_diff, col_pt = calculate_liveliness(ego_state, self.opp_state)
        # self.liveliness.append(l)
        #print("livesness", l)
        self.liveliness.append((l, pos_diff, vel_diff))
        self.u_ori.append(u1.ravel())
        if col_pt and l < config.liveness_threshold:
            self.last_liveliness_iteration = self.env.sim_iteration
            self.col_pt = col_pt

        if not mpccbf:
            if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR and config.liveliness:
                #if self.agent_idx == 0 and self.env.sim_iteration < self.last_liveliness_iteration + 5:
                if self.env.sim_iteration < self.last_liveliness_iteration + 2:
                    u1_before_proj = u1.copy()
                    # curr_v0_v1_point = np.array([0.0, 0.0])
                    # curr_v0_v1_point[self.agent_idx] = ego_state[3]
                    # curr_v0_v1_point[1 - self.agent_idx] = self.opp_state[3]
                    # desired_v0_v1_vec = np.array([1, config.zeta])
                    # desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)
                    # desired_v0_v1_point = np.dot(curr_v0_v1_point, desired_v0_v1_vec_normalized) * desired_v0_v1_vec_normalized
                    # mult_factor = (desired_v0_v1_point[self.agent_idx]) / u1[0]
                    ego_dist = np.linalg.norm(ego_state[0:2] - np.array(self.col_pt))
                    opp_dist = np.linalg.norm(self.opp_state[0:2] - np.array(self.col_pt))
                    
                    if not self.role and ego_dist > opp_dist or self.role==Role.FOLLOWER:
                        opp_ttc = (opp_dist + 0.25) / config.v_limit
                        #opp_ttc = opp_dist+2 / self.opp_state[3]
                        new_vel = (ego_dist - 0.25 ) / opp_ttc
                        #print(self.opp_state[0:2], np.array(self.col_pt), self.opp_state[0:2] - np.array(self.col_pt))
                        #print(self.col_pt, self.opp_state, opp_dist, opp_ttc, ego_dist, new_vel)
                        u1[0] = new_vel
                        u1[0] = min(new_vel, config.v_limit)
                    else:
                        u1[0] = config.v_limit

        else:
            if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR and config.liveliness:
                if self.env.sim_iteration < self.last_liveliness_iteration + 5:
                    u1_before_proj = u1.copy()
                    curr_v0_v1_point = np.array([0.0, 0.0])
                    curr_v0_v1_point[self.agent_idx] = ego_state[3]
                    curr_v0_v1_point[1 - self.agent_idx] = self.opp_state[3]

                    ego_dist = np.linalg.norm(ego_state[0:2] - np.array(self.col_pt))
                    opp_dist = np.linalg.norm(self.opp_state[0:2] - np.array(self.col_pt))
                    if not self.role and ego_dist > opp_dist or self.role==Role.FOLLOWER:
                        if self.agent_idx == 0:
                            desired_v0_v1_vec = np.array([1, config.zeta])
                        else:
                            desired_v0_v1_vec = np.array([config.zeta, 1])
                        # desired_v0_v1_vec = np.array([1, config.zeta])
                        desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)
                        desired_v0_v1_point = np.dot(curr_v0_v1_point, desired_v0_v1_vec_normalized) * desired_v0_v1_vec_normalized
                        mult_factor = (desired_v0_v1_point[self.agent_idx]) / u1[0]
                    else:
                        mult_factor = 1
                    
                    
                    u1[0] *= mult_factor
                        
                    u1[0] = min(u1[0], config.v_limit)
                # else:
                #         u1[0] = config.v_limit


                
                # u1[0] *= mult_factor
                #u1 *= mult_factor

                #u1[0] = min(u1[0], config.v_limit)
                
                # print(f"Running liveness {l}")
                # #print("mult factor", mult_factor)
                # print("Position diff:", pos_diff)
                # print("Velocity diff:", vel_diff)
                # print(f"\tEgo Vel: {ego_state[3]}, Opp Vel: {self.opp_state[3]}")
                # #print(f"\tP1: {curr_v0_v1_point}, Desired P1: {desired_v0_v1_point}.")
                # print(f"Original control {u1_before_proj.T}. Output control {u1.T}")

                # v = (xf_minus_two - xf_two)/T
                # norm_u1 = np.linalg.norm(u1_before_proj)
                # norm_v = np.linalg.norm(v)

                # # Special case: if u is the zero vector, return any point on the circle of radius half_norm_v   
                # if np.allclose(u1_before_proj, np.zeros_like(u1_before_proj)):
                #     return np.array([norm_v / 2.5, 0])

                # u = (u1_before_proj / norm_u1) * (norm_u1 / 2)
                # u1 = u

        return u1
