import math
import do_mpc
from casadi import *
import config
from config import DynamicsModel
import numpy as np
from util import calculate_all_metrics, get_ray_intersection_point
from memory_profiler import profile

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
    def __init__(self, agent_idx, opp_gamma, obs_gamma, live_gamma, liveness_thresh, goal, static_obs = [], delay_start = 0.0):
        self.agent_idx = agent_idx
        self.goal = goal
        self.static_obs = static_obs
        self.delay_start = delay_start
        self.opp_state = None
        self.opp_gamma = opp_gamma
        self.obs_gamma = obs_gamma
        self.live_gamma = live_gamma
        self.liveness_thresh = liveness_thresh
        self.Q = config.COST_MATRICES[config.dynamics]['Q']
        self.R = config.COST_MATRICES[config.dynamics]['R']

    def initialize_controller(self, env):
        self.model = env.define_model(call_setup = False)
        self.model.set_variable('_tvp', 'x_moving_obs')
        self.model.set_variable('_tvp', 'y_moving_obs')
        self.model.setup()
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
                     't_step': config.MPC_Ts,
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
            max_u = np.array([config.omega_limit, config.accel_limit])
            mpc.bounds['lower', '_u', 'u'] = -max_u
            mpc.bounds['upper', '_u', 'u'] = max_u

            min_x = np.array([-float("inf"), -float("inf"), -float("inf"), 0.0])
            max_x = np.array([float("inf"), float("inf"), float("inf"), config.v_limit])
            mpc.bounds['lower', '_x', 'x'] = min_x
            mpc.bounds['upper', '_x', 'x'] = max_x
        
        mpc = self.set_tvp_for_mpc(mpc)

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
        x_k1 = self.model.x['x'] + A*config.MPC_Ts + B@self.model.u['u']*config.MPC_Ts

        # Compute CBF constraints
        cbf_constraints = []
        for obs in self.static_obs:
            # delta_h_k + gamma*h_k >= 0
            # h_k1 - h_k + gamma*h_k >= 0
            # -h_k1 + h_k - gamma*h_k <= 0
            # -h_k1 + (1 - gamma)*h_k <= 0
            h_k = self.h_obs(self.model.x['x'], obs)
            h_k1 = self.h_obs(x_k1, obs)
            cbf_constraints.append(-h_k1 + (1-self.obs_gamma)*h_k)

        if config.mpc_use_opp_cbf:
            # obs = (self.model.tvp['x_moving_obs'], self.model.tvp['y_moving_obs'], config.agent_radius)
            obs = (self.opp_state[0], self.opp_state[1], config.agent_radius)
            obs_projected = (self.opp_state[0] + self.opp_state[3] * math.cos(self.opp_state[2]) * config.MPC_Ts, self.opp_state[1] + self.opp_state[3] * math.sin(self.opp_state[2]) * config.MPC_Ts, config.agent_radius)
            h_k = self.h_obs(self.model.x['x'], obs)
            # h_k1 = self.h_obs(x_k1, obs)
            h_k1 = self.h_obs(x_k1, obs_projected)
            # print(obs, obs_projected)
            cbf_constraints.append(-h_k1 + (1-self.opp_gamma)*h_k)

        return cbf_constraints
    
    """Computes the Control Barrier Function for an obstacle."""
    def h_obs(self, x, obstacle):
        x_obs, y_obs, r_obs = obstacle
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (config.agent_radius + r_obs + config.safety_dist)**2
        return h
    
    def set_tvp_for_mpc(self, mpc):
        """Sets the trajectory for trajectory tracking and/or the moving obstacles' trajectory.

        Inputs:
          - mpc(do_mpc.controller.MPC): The mpc controller
        Returns:
          - mpc(do_mpc.controller.MPC): The mpc model with time-varying parameters added
        """
        tvp_struct_mpc = mpc.get_tvp_template()

        def tvp_fun_mpc(t_now):
            # Moving obstacles trajectory
            tvp_struct_mpc['_tvp', :, 'x_moving_obs'] = self.opp_state[0] + self.opp_state[3] * math.cos(self.opp_state[2]) * t_now
            tvp_struct_mpc['_tvp', :, 'y_moving_obs'] = self.opp_state[1] + self.opp_state[3] * math.sin(self.opp_state[2]) * t_now

            return tvp_struct_mpc

        mpc.set_tvp_fun(tvp_fun_mpc)
        return mpc

    # Assumes that the double-integrator dynamic model is being used
    def add_liveliness_constraint(self, mpc):
        if self.opp_state is None:
            return

        l, _, _, _, intersecting, is_live = calculate_all_metrics(self.initial_state.copy(), self.opp_state, self.liveness_thresh)
        # if is_live:
        if not intersecting:
            return

        # print(f"Adding constraint, liveliness = {l}, intersecting = {intersecting}")

        # Get state vector x_{t+k+1}
        A, B = self.env.get_dynamics(self.model.x['x'])
        x_k1 = self.model.x['x'] + A*config.MPC_Ts + B@self.model.u['u']*config.MPC_Ts

        # Compute CBF constraints
        opp_state = np.array(self.opp_state)
        opp_A, _ = self.env.get_double_integrator_dynamics_np(opp_state)

        h_k = self.h_v(self.model.x['x'], self.opp_state, ts=0.0, t=0)

        opp_k1 = opp_state + opp_A*config.MPC_Ts
        h_k1 = self.h_v(x_k1, opp_k1, ts=config.MPC_Ts, t=1)
        constraint = -h_k1 + (1-self.live_gamma)*h_k
        # -h_k1 + (1 - gamma)*h_k <= 0
        # h_k1 >= h_k - gamma*h_k
        # (h_k1 - h_k) >= -gamma*h_k
        mpc.set_nl_cons('liveliness_constraint', constraint, ub=0)

    # def h_v(self, x, opp_x):
    #     self.A_matrix = SX.zeros(2, 2)
    #     max_zeta = 0.3 / opp_x[3]
    #     upper_zeta = min(max_zeta, config.zeta)

    #     # The A matrix looks like this:
    #     # [[1, -zeta]
    #     #  [-zeta, 1]]
    #     # So that when multiplied by the velocity vector [ego_v, opp_v], the result is [ego_v - zeta*opp_v, opp_v - zeta*ego_v]
    #     self.A_matrix[0, 0] = 1.0
    #     self.A_matrix[0, 1] = -upper_zeta
    #     self.A_matrix[1, 0] = -config.zeta
    #     self.A_matrix[1, 1] = 1.0

    #     # ego_v - 3 * opp_v >= 0.0 -> ego_v >= 3 * opp_v
    #     # opp_v - 3 * ego_v >= 0.0 -> ego_v <= 1/3 * opp_v
    #     # Means that agent 1 will speed up and agent 2 will slow down.

    #     vel_vector = vertcat(x[3], opp_x[3])
    #     h_vec = self.A_matrix @ vel_vector
    #     # If agent 0 should go faster, then h_idx = self.agent_idx, otherwise h_idx = 1 - self.agent_idx
    #     h_idx = self.agent_idx if config.mpc_p0_faster else 1 - self.agent_idx
    #     h = h_vec[h_idx]
    #     return h

    # def h_v(self, x, opp_state, ts, t):
    #     intersection = get_ray_intersection_point(self.initial_state[:2], self.initial_state[2], opp_state[:2], opp_state[2])
    #     assert(intersection is not None)

    #     # d0 = sqrt((self.initial_state[0] - intersection[0])**2 + (self.initial_state[1] - intersection[1])**2)
    #     d0 = sqrt((x[0] - intersection[0]) ** 2.0 + (x[1] - intersection[1]) ** 2.0)
    #     d0_reg = np.sqrt((self.initial_state[0] - intersection[0]) ** 2.0 + (self.initial_state[1] - intersection[1]) ** 2.0)
    #     d1 = np.linalg.norm(opp_state[:2] - intersection)
    #     should_go_faster = (config.mpc_p0_faster and self.agent_idx == 0) or (not config.mpc_p0_faster and self.agent_idx == 1)

    #     theta = np.abs(opp_state[2] - self.initial_state[2])
    #     if theta >= 2.0 * np.pi:
    #         theta -= 2.0 * np.pi
    #     if theta > np.pi:
    #         theta = 2.0 * np.pi - theta

    #     d0 = d0_reg
    #     d0 -= x[3]*ts
    #     d0_reg -= self.initial_state[3]*ts

    #     critical_point = config.agent_radius / np.sin(theta)
    #     d0 -= critical_point
    #     d1 -= critical_point
    #     d0_reg -= critical_point

    #     if should_go_faster:
    #         d0 += config.agent_radius + config.safety_dist
    #         d1 -= (config.agent_radius + config.safety_dist)
    #         # h = (d1 * x[3] - d0 * opp_state[3])
    #         h = (d1 * x[3] - d0_reg * opp_state[3])
    #     else:
    #         if t == 0:
    #             print("Intersection:", intersection)
    #             print(f"D0 orig: {d0_reg}, D1 orig: {d1}, Critical dist: {critical_point}")
    #         d0 -= (config.agent_radius + config.safety_dist)
    #         d0_reg -= (config.agent_radius + config.safety_dist)
    #         d1 += config.agent_radius + config.safety_dist
    #         # h = (d0 * opp_state[3] - d1 * x[3])
    #         h = (d0_reg * opp_state[3] - d1 * x[3])
    #         if t == 0:
    #             print(f"D0 perturb: {d0_reg}, D1 perturb: {d1}")
    #             print("CBF:", d0_reg * opp_state[3] - d1 * self.initial_state[3])

    #     # if should_go_faster:
    #     #     h = d1 - 2 * config.agent_radius - config.safety_dist - d0
    #     # else:
    #     #     h = d0 - 2 * config.agent_radius - config.safety_dist - d1
    #     #     print(d0, d1, self.initial_state[3], opp_state[3], np.degrees(self.initial_state[2]), np.degrees(opp_state[2]), h)
    #     #     print("CBF:", d0_reg - 2 * config.agent_radius - config.safety_dist - d1)

    #     return h

    def h_v(self, x, opp_state, ts, t):
        dir_to_opp = np.arctan2(opp_state[1] - self.initial_state[1], opp_state[0] - self.initial_state[0])
        vec_to_opp = np.array([np.cos(dir_to_opp), np.sin(dir_to_opp)])
        initial_closest_to_opp = np.array(self.initial_state[:2]) + vec_to_opp * (config.agent_radius + config.mpc_liveness_safety_buffer / 2.0)
        opp_closest_to_initial = np.array(self.opp_state[:2]) - vec_to_opp * (config.agent_radius + config.mpc_liveness_safety_buffer / 2.0)
        intersection = get_ray_intersection_point(initial_closest_to_opp, self.initial_state[2], opp_closest_to_initial, opp_state[2])
        assert(intersection is not None)

        # d0 = sqrt((x[0] - intersection[0]) ** 2.0 + (x[1] - intersection[1]) ** 2.0)
        d0_reg = np.linalg.norm(initial_closest_to_opp - intersection)
        d1 = np.linalg.norm(opp_closest_to_initial - intersection)
        should_go_faster = (config.mpc_p0_faster and self.agent_idx == 0) or (not config.mpc_p0_faster and self.agent_idx == 1)

        d0 = d0_reg
        d0 -= x[3]*ts
        d0_reg -= self.initial_state[3]*ts

        if should_go_faster:
            h = (d1 * x[3] - d0 * opp_state[3])
        else:
            if t == 0:
                print("Closest points:", initial_closest_to_opp, opp_closest_to_initial)
                print("Intersection:", intersection)
                print(f"D0 orig: {d0_reg}, D1 orig: {d1}")

            h = (d0 * opp_state[3] - d1 * x[3])
            if t == 0:
                print("CBF:", d0_reg * opp_state[3] - d1 * self.initial_state[3])

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

    # @profile
    def make_step(self, timestamp, x0):
        if timestamp < self.delay_start:
            self.use_for_training = False
            return np.zeros((config.num_controls, 1))
        self.use_for_training = True

        u1 = self.mpc.make_step(x0)

        # Add liveliness condition here
        ego_state = self.initial_state.copy()
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            ego_state = np.append(ego_state, [u1[0][0]])

        return u1
