import do_mpc
from casadi import *
import config
from config import DynamicsModel
import numpy as np
from util import calculate_all_metrics

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
    def __init__(self, agent_idx, goal, static_obs = [], delay_start = 0.0):
        self.agent_idx = agent_idx
        self.goal = goal
        self.static_obs = static_obs
        self.delay_start = delay_start
        self.opp_state = None
        self.Q = config.COST_MATRICES[config.dynamics]['Q']
        self.R = config.COST_MATRICES[config.dynamics]['R']

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
            max_u = np.array([config.accel_limit, config.omega_limit])
            mpc.bounds['lower', '_u', 'u'] = -max_u
            mpc.bounds['upper', '_u', 'u'] = max_u

            min_x = np.array([-float("inf"), -float("inf"), -float("inf"), 0.0])
            max_x = np.array([float("inf"), float("inf"), float("inf"), config.v_limit])
            mpc.bounds['lower', '_x', 'x'] = min_x
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
        x_k1 = self.model.x['x'] + A*config.MPC_Ts + B@self.model.u['u']*config.MPC_Ts

        # Compute CBF constraints
        cbf_constraints = []
        opponent_obs = (self.opp_state[0], self.opp_state[1], config.agent_radius)
        for obs in self.static_obs + [opponent_obs]:
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

        l, _, _, _, intersecting = calculate_all_metrics(self.initial_state.copy(), self.opp_state)
        if l > config.liveness_threshold or not intersecting:
            return
        # if l > config.liveness_threshold:
        #     return

        print(f"Adding constraint, liveliness = {l}, intersecting = {intersecting}")

        # Get state vector x_{t+k+1}
        A, B = self.env.get_dynamics(self.model.x['x'])
        x_k1 = self.model.x['x'] + A*config.MPC_Ts + B@self.model.u['u']*config.MPC_Ts

        # Compute CBF constraints
        h_k = self.h_v(self.model.x['x'], self.opp_state)
        h_k1 = self.h_v(x_k1, self.opp_state)
        constraint = -h_k1 + (1-config.liveliness_gamma)*h_k
        # -h_k1 + (1 - gamma)*h_k <= 0
        # h_k1 >= h_k - gamma*h_k
        # (h_k1 - h_k) >= -gamma*h_k
        mpc.set_nl_cons('liveliness_constraint', constraint, ub=0)

    def h_v(self, x, opp_x):
        self.A_matrix = SX.zeros(2, 2)
        max_zeta = 0.3 / opp_x[3]
        upper_zeta = min(max_zeta, config.zeta)

        # The A matrix looks like this:
        # [[1, -zeta]
        #  [-zeta, 1]]
        # So that when multiplied by the velocity vector [ego_v, opp_v], the result is [ego_v - zeta*opp_v, opp_v - zeta*ego_v]
        self.A_matrix[0, 0] = 1.0
        self.A_matrix[0, 1] = -upper_zeta
        self.A_matrix[1, 0] = -config.zeta
        self.A_matrix[1, 1] = 1.0

        # ego_v - 3 * opp_v >= 0.0 -> ego_v >= 3 * opp_v
        # opp_v - 3 * ego_v >= 0.0 -> ego_v <= 1/3 * opp_v
        # Means that agent 1 will speed up and agent 2 will slow down.
    
        vel_vector = vertcat(x[3], opp_x[3])
        h_vec = self.A_matrix @ vel_vector
        # If agent 0 should go faster, then h_idx = self.agent_idx, otherwise h_idx = 1 - self.agent_idx
        h_idx = self.agent_idx if config.mpc_p0_faster else 1 - self.agent_idx
        h = h_vec[h_idx]
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

    def make_step(self, timestamp, x0):
        if timestamp < self.delay_start:
            return np.zeros((config.num_controls, 1))

        u1 = self.mpc.make_step(x0)

        # Add liveliness condition here
        ego_state = self.initial_state.copy()
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            ego_state = np.append(ego_state, [u1[0][0]])

        return u1
