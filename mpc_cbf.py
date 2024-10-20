import do_mpc
from casadi import *
import config
from config import DynamicsModel
from numpy import linalg as LA
import numpy as np

EPSILON = 0.001

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
    def __init__(self, agent_idx, initial_state, goal, static_obs = [], opp_state = None):
        self.agent_idx = agent_idx
        self.initial_state = initial_state
        self.goal = goal
        self.static_obs = static_obs
        self.opp_state = opp_state
        self.Q = config.COST_MATRICES[config.dynamics]['Q']
        self.R = config.COST_MATRICES[config.dynamics]['R']

        self.model = self.define_model()
        self.mpc = self.define_mpc()
        self.simulator = self.define_simulator()

    def define_model(self):
        """Configures the dynamical model of the system (and part of the objective function).

        x_{k+1} = x_k + B*u_k*T_s
        Returns:
          - model(do_mpc.model.Model): The system model
        """

        model_type = 'discrete'
        model = do_mpc.model.Model(model_type)

        # Num states, State Space matrices, and Input Space matrices
        _x = model.set_variable(var_type='_x', var_name='x', shape=(config.num_states, 1))
        _u = model.set_variable(var_type='_u', var_name='u', shape=(config.num_controls, 1))
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            A, B = self.get_single_integrator_dynamics(_x)
        else:
            A, B = self.get_double_integrator_dynamics(_x)

        # Set right-hand-side of ODE for all introduced states (_x).
        x_next = _x + A*config.Ts + B@_u*config.Ts
        model.set_rhs('x', x_next, process_noise=False)  # Set to True if adding noise

        # Setup model
        model.setup()
        return model

    """Defines the system input matrices A and B for single-integrator dynamics."""
    @staticmethod
    def get_single_integrator_dynamics(x):
        A = SX.zeros(3, 1)

        a = 1e-9  # Small positive constant so system has relative degree 1
        # [[cos(theta), 0]
        #  [sin(theta), 0]
        #  [0,          1]]
        B = SX.zeros(3, 2)
        B[0, 0] = cos(x[2])
        B[0, 1] = -a*sin(x[2])
        B[1, 0] = sin(x[2])
        B[1, 1] = a*cos(x[2])
        B[2, 1] = 1
        return A, B

    """Defines the system input matrices A and B for single-integrator dynamics."""
    @staticmethod
    def get_double_integrator_dynamics(x):
        A = SX.zeros(4, 1)
        A[0] = x[3] * cos(x[2]) # x_dot = v * cos(theta)
        A[1] = x[3] * sin(x[2]) # y_dot = v * sin(theta)

        B = SX.zeros(4, 2)
        B[3, 0] = 1 # dv = a
        B[2, 1] = 1 # dtheta = omega

        a = 1e-9  # Small positive constant so system has relative degree 1
        B[0, 1] = -a*sin(x[2])
        B[1, 1] = a*cos(x[2])
        return A, B

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
                     # 'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
                     }
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
        self.add_cbf_constraints(mpc)

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
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            A, B = self.get_single_integrator_dynamics(self.model.x['x'])
        else:
            A, B = self.get_double_integrator_dynamics(self.model.x['x'])

        x_k1 = self.model.x['x'] + A*config.Ts + B@self.model.u['u']*config.Ts

        # Compute CBF constraints
        cbf_constraints = []
        for obs in self.static_obs:
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

        ego_state = self.initial_state.copy()
        ego_vel = np.array([ego_state[3] * np.cos(ego_state[2]), ego_state[3] * np.sin(ego_state[2])])
        opp_vel = np.array([self.opp_state[3] * np.cos(self.opp_state[2]), self.opp_state[3] * np.sin(self.opp_state[2])])
        vel_diff = ego_vel - opp_vel
        pos_diff = ego_state[:2] - self.opp_state[:2]
        l = np.arccos(abs(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
        if l > config.liveness_threshold:
            return
        
        print(f"Adding constraint, liveliness = {l}")

        # Get state vector x_{t+k+1}
        A, B = self.get_double_integrator_dynamics(self.model.x['x'])
        x_k1 = self.model.x['x'] + A*config.Ts + B@self.model.u['u']*config.Ts

        # Compute CBF constraints
        h_k = self.h_v(self.model.x['x'], self.opp_state)
        h_k1 = self.h_v(x_k1, self.opp_state)
        constraint = -h_k1 + (1-config.liveliness_gamma)*h_k
        # -h_k1 + (1 - gamma)*h_k <= 0
        # h_k1 >= h_k - gamma*h_k
        # (h_k1 - h_k) >= -gamma*h_k
        print(constraint)
        mpc.set_nl_cons('liveliness_constraint', constraint, ub=0)


    def h_v(self, x, opp_x):
        self.A_matrix = SX.zeros(2, 2)
        self.A_matrix[0, 0] = 1.0
        self.A_matrix[0, 1] = -config.zeta
        self.A_matrix[1, 0] = -config.zeta
        self.A_matrix[1, 1] = 1.0

        # ego_v - 3 * opp_v >= 0.0 -> ego_v >= 3 * opp_v
        # opp_v - 3 * ego_v >= 0.0 -> ego_v <= 1/3 * opp_v
        # Means that agent 1 will speed up and agent 2 will slow down.

        vel_vector = vertcat(x[3], opp_x[3])
        print(vel_vector)
        h_vec = self.A_matrix @ vel_vector
        h = h_vec[self.agent_idx]
        # h = mmax(h_vec)
        return h

    """Configures the simulator."""
    def define_simulator(self):
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=config.Ts)
        simulator.setup()

        return simulator

    """Sets the initial state in all components."""
    def set_init_state(self, x0):
        self.mpc.setup()
        self.mpc.reset_history()
        self.simulator.reset_history()
        self.mpc.x0 = x0
        self.mpc.u0 = np.zeros_like(self.mpc.u0['u'])
        self.simulator.x0 = x0
        self.mpc.set_initial_guess()

    def run_simulation_to_get_final_condition(self,x0,xff,j):
        """Runs a closed-loop control simulation."""
        c=j*2+self.agent_idx
        ego_xf=xff[c,0:2] 
        opp_xf=xff[c-1,0:2]
        opp_xf_prev_iter=xff[c-3,0:2]

        u1 = self.mpc.make_step(x0)
        u1_before_proj=u1.copy()

        # Add liveliness condition here
        l = 0.0
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR and config.liveliness:
            # If liveness is turned on, check for liveliness condition and adjust our control input if needed.
            ego_vel = np.array([u1[0][0] * np.cos(xff[c,2]), u1[0][0] * np.sin(xff[c,2])])
            opp_vel = (opp_xf - opp_xf_prev_iter)/config.Ts
            vel_diff = ego_vel - opp_vel
            pos_diff = ego_xf - opp_xf
            l=np.arccos(abs(np.dot(vel_diff,pos_diff))/(LA.norm(vel_diff)*LA.norm(pos_diff)+EPSILON))
            if j>3 and self.agent_idx == 1 and l < config.liveness_threshold:
                v_ego, v_opp = np.linalg.norm(ego_vel), np.linalg.norm(opp_vel)
                curr_v0_v1_point = np.array([0.0, 0.0])
                curr_v0_v1_point[self.agent_idx] = v_ego
                curr_v0_v1_point[1 - self.agent_idx] = v_opp
                desired_v0_v1_vec = np.array([config.zeta, 1.0])
                desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)
                desired_v0_v1_point = np.dot(curr_v0_v1_point, desired_v0_v1_vec_normalized) * desired_v0_v1_vec_normalized
                mult_factor = (desired_v0_v1_point[self.agent_idx]*config.Ts) / u1[0]
                u1 *= mult_factor
                print(f"Running liveness {l}")
                print("Position diff:", pos_diff)
                print("Velocity diff:", vel_diff)
                print(f"\tEgo Vel: {ego_vel}, Opp Vel: {opp_vel}")
                print(f"\tP1: {curr_v0_v1_point}, Desired P1: {desired_v0_v1_point}.")
                print(f"Original control {u1_before_proj.T}. Output control {u1.T}")

                # v = (xf_minus_two - xf_two)/T
                # norm_u1 = np.linalg.norm(u1_before_proj)
                # norm_v = np.linalg.norm(v)

                # # Special case: if u is the zero vector, return any point on the circle of radius half_norm_v   
                # if np.allclose(u1_before_proj, np.zeros_like(u1_before_proj)):
                #     return np.array([norm_v / 2.5, 0])

                # u = (u1_before_proj / norm_u1) * (norm_u1 / 2)
                # u1 = u

        x1 = self.simulator.make_step(u1)

        return x1, u1_before_proj, u1, l
