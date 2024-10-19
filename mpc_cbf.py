import do_mpc
from casadi import *
import config
from config import DynamicsModel
from numpy import linalg as LA
import numpy as np


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
    def __init__(self, goal, static_obs = []):
        self.sim_time = config.sim_time          # Total simulation time steps
        self.Q = config.COST_MATRICES[config.dynamics]['Q']
        self.R = config.COST_MATRICES[config.dynamics]['R']
        self.static_obs = static_obs
        self.goal = goal

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

        # Optional: Define an expression, which represents the stage and terminal
        # cost of the control problem. This term will be later used as the cost in
        # the MPC formulation and can be used to directly plot the trajectory of
        # the cost of each state.
        cost_expr = self.get_cost_expression(model)
        model.set_expression(expr_name='cost', expr=cost_expr)

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

        a = 1e-9  # Small positive constant so system has relative degree 1
        B = SX.zeros(4, 2)
        B[3, 0] = 1 # dv = a
        B[2, 1] = 1 # dtheta = omega
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
        mterm = self.model.aux['cost']  # Terminal cost
        lterm = self.model.aux['cost']  # Stage cost
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
            h_k1 = self.h(x_k1, obs)
            h_k = self.h(self.model.x['x'], obs)
            cbf_constraints.append(-h_k1 + (1-config.gamma)*h_k)

        return cbf_constraints

    """Computes the Control Barrier Function for an obstacle."""
    def h(self, x, obstacle):
        x_obs, y_obs, r_obs = obstacle
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (config.agent_radius + r_obs + config.safety_dist)**2
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

    def run_simulation(self,x0):
        """Runs a closed-loop control simulation."""
        # print("Simulator at beginning of simulation 1", self.simulator.x0)
        for k in range(self.sim_time):
            u0 = self.mpc.make_step(x0)
            x0 = self.simulator.make_step(u0)
            # y_next = self.simulator.make_step(u0, w0=10**(-4)*np.random.randn(3, 1))  # Optional Additive process noise
        # print("Simulator at end of simulation 1", self.simulator.x0)
        return

    def run_simulation_to_get_final_condition(self,x0,xff,j,i):
        """Runs a closed-loop control simulation."""
        x1 = x0
        T=0.1
        epsilon=0.001
        # NOTE: For j < 3 the liveness value will be accurate lowkey.
        c=j*2+i
        # print("C", c)
        xf_minus_one=xff[c,0:2] 
        xf_one=xff[c-2,0:2]
        xf_minus_two=xff[c-1,0:2]
        xf_two=xff[c-3,0:2]

        # xf_minus_one=xff[j,0:2] 
        # xf_one=xff[j-2,0:2]
        # xf_minus_two=xff[j-1,0:2]
        # xf_two=xff[j-3,0:2]

        # Will add liveliness condition here
        vec1=((xf_minus_two-xf_two)-(xf_minus_one-xf_one))/T#((xf_minus[j,0:2]-xf[j,0:2])-(xf_minus[i,0:2]-xf[i,0:2]))/T
        vec2=(xf_minus_two - xf_minus_one)#xf[j,0:2]-xf[i,0:2]
        l=np.arccos(abs(np.dot(vec1,vec2))/(LA.norm(vec1)*LA.norm(vec2)+epsilon))
        # print("X1 at beginning of final simulation", x1)
        # print("Simulator at beginning of final simulation", self.simulator.x0['x'])
        for k in range(self.sim_time):
            u1 = self.mpc.make_step(x1)
            u1_before_proj=u1.copy()
            if j>3 and i == 1 and config.dynamics == DynamicsModel.SINGLE_INTEGRATOR and config.liveliness and l < config.liveness_threshold:
                print("YOOOO")
                # v_ego = u1[0] / T
                # v_opp = np.linalg.norm((xf_minus_two - xf_two))/(T*4)
                # curr_v0_v1_point = np.array([0.0, 0.0])
                # curr_v0_v1_point[i] = v_ego
                # curr_v0_v1_point[1 - i] = v_opp
                # desired_v0_v1_vec = np.array([3.0, 1.0])
                # desired_v0_v1_vec_normalized = desired_v0_v1_vec / np.linalg.norm(desired_v0_v1_vec)
                # desired_v0_v1_point = np.dot(curr_v0_v1_point, desired_v0_v1_vec_normalized) * desired_v0_v1_vec_normalized
                # mult_factor = (desired_v0_v1_point[i]*T) / u1[0]
                # u1 *= mult_factor
                # print(f"Running liveness {l}. Original control {u1_before_proj.T}. Output control {u1.T}")
                # print(f"\tEgo Points: {xf_one}, {xf_minus_one}")
                # print(f"\tOpp Points: {xf_two}, {xf_minus_two}")
                # print(f"\tEgo Vel: {v_ego}, Opp Vel: {v_opp}")
                # print(f"\tP1: {curr_v0_v1_point}, Desired P1: {desired_v0_v1_point}.")
                # print(f"\tdVel Vec: {vec1}, dPos Vec: {vec2}, L: {l}")

                v = (xf_minus_two - xf_two)/T
                norm_u1 = np.linalg.norm(u1_before_proj)
                norm_v = np.linalg.norm(v)

                # Special case: if u is the zero vector, return any point on the circle of radius half_norm_v   
                if np.allclose(u1_before_proj, np.zeros_like(u1_before_proj)):
                    return np.array([norm_v / 2.5, 0])

                u = (u1_before_proj / norm_u1) * (norm_u1 / 2)
                u1 = u

            # Calculate the stage cost for each timestep
            # Below is the game theoretic control input chosen
            # if l<0.2 and LA.norm(vec2)<0.2 and np.matmul(A,u1)<0:
            #     u1=np.matmul(inv(A),u1)
            # print(f"\tU1 before proj: {u1_before_proj.T}, U1: {u1.T}")
            x1 = self.simulator.make_step(u1)
            # if k == 0:
            #     print("Simualtor after step 1 of simulation 2", self.simulator.x0)
            # y_next = self.simulator.make_step(u0, w0=10**(-4)*np.random.randn(3, 1))  # Optional Additive process noise
            # print(f"\tX1: {x1.T}")
        # print("Simualtor at end of final simulation", x1)
        # print("Control at end of final simulation", u1)
        return x1, u1_before_proj, u1, l
