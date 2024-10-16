import do_mpc
from casadi import *
import config
from numpy import linalg as LA
import casadi as ca
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
    def __init__(self):
        self.sim_time = config.sim_time          # Total simulation time steps
        self.Ts = config.Ts                      # Sampling time
        self.R = config.R                        # Controls cost matrix
        self.Q = config.Q  
        self.A = np.array([[1, -2],[-2, 1]] )                  # State cost matrix
        self.static_obstacles_on = config.static_obstacles_on  # Whether to have static obstacles
        self.moving_obstacles_on = config.moving_obstacles_on  # Whether to have moving obstacles
        if self.static_obstacles_on:
            self.obs = config.obs                # Static Obstacles
        if self.moving_obstacles_on:             # Moving obstacles
            self.moving_obs = config.moving_obs
        self.r = config.r                        # Robot radius
        self.control_type = config.control_type  # "setpoint" or "traj_tracking"
        if self.control_type == "setpoint":      # Go-to-goal
            self.goal = config.goal              # Robot's goal pose
        self.gamma = config.gamma                # CBF parameter
        self.safety_dist = config.safety_dist    # Safety distance

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

        # States: (x, y, theta)
        n_states = 3
        _x = model.set_variable(var_type='_x', var_name='x', shape=(n_states, 1))

        # Inputs: (v, omega)
        n_controls = 2
        _u = model.set_variable(var_type='_u', var_name='u', shape=(n_controls, 1))

        # State Space matrices
        B = self.get_sys_matrix_B(_x)

        # Set right-hand-side of ODE for all introduced states (_x).
        x_next = _x + B@_u*self.Ts
        model.set_rhs('x', x_next, process_noise=False)  # Set to True if adding noise

        # Optional: Define an expression, which represents the stage and terminal
        # cost of the control problem. This term will be later used as the cost in
        # the MPC formulation and can be used to directly plot the trajectory of
        # the cost of each state.
        model, cost_expr = self.get_cost_expression(model)
        model.set_expression(expr_name='cost', expr=cost_expr)

        # Moving obstacle (define time-varying parameter for its position)
        if self.moving_obstacles_on is True:
            for i in range(len(self.moving_obs)):
                model.set_variable('_tvp', 'x_moving_obs'+str(i))
                model.set_variable('_tvp', 'y_moving_obs'+str(i))

        # Setup model
        model.setup()
        return model

    @staticmethod
    def get_sys_matrix_B(x):
        """Defines the system input matrix B.

        Inputs:
          - x(casadi.casadi.SX): The state vector [3x1]
        Returns:
          - B(casadi.casadi.SX): The system input matrix B [3x2]
        """
        a = 1e-9  # Small positive constant so system has relative degree 1
        B = SX.zeros(3, 2)
        B[0, 0] = cos(x[2])
        B[0, 1] = -a*sin(x[2])
        B[1, 0] = sin(x[2])
        B[1, 1] = a*cos(x[2])
        B[2, 1] = 1
        return B

    def get_cost_expression(self, model):
        """Defines the objective function wrt the state cost depending on the type of control.

        Inputs:
          - model(do_mpc.model.Model):         The system model
        Returns:
          - model(do_mpc.model.Model):         The system model
          - cost_expression(casadi.casadi.SX): The objective function cost wrt the state
        """

        if self.control_type == "setpoint":  # Go-to-goal
            # Define state error
            X = model.x['x'] - self.goal
        else:                                # Trajectory tracking
            # Set time-varying parameters for the objective function
            model.set_variable('_tvp', 'x_set_point')
            model.set_variable('_tvp', 'y_set_point')

            # Define state error
            theta_des = np.arctan2(model.tvp['y_set_point'] - model.x['x', 1], model.tvp['x_set_point'] - model.x['x', 0])
            X = SX.zeros(3, 1)
            X[0] = model.x['x', 0] - model.tvp['x_set_point']
            X[1] = model.x['x', 1] - model.tvp['y_set_point']
            X[2] = np.arctan2(sin(theta_des - model.x['x', 2]), cos(theta_des - model.x['x', 2]))

        cost_expression = transpose(X)@self.Q@X
        return model, cost_expression

    def define_mpc(self):
        """Configures the mpc controller.

        Returns:
          - mpc(do_mpc.model.MPC): The mpc controller
        """

        mpc = do_mpc.controller.MPC(self.model)

        # Set parameters
        setup_mpc = {'n_robust': 0,  # Robust horizon
                     'n_horizon': config.T_horizon,
                     't_step': self.Ts,
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
        max_u = np.array([config.v_limit, config.omega_limit])
        mpc.bounds['lower', '_u', 'u'] = -max_u
        mpc.bounds['upper', '_u', 'u'] = max_u

        # If trajectory tracking or moving obstacles: Define time-varying parameters
        if self.control_type == "traj_tracking" or self.moving_obstacles_on is True:
            mpc = self.set_tvp_for_mpc(mpc)

        # Add safety constraints
        if self.static_obstacles_on or self.moving_obstacles_on:
            # MPC-CBF: Add CBF constraints
            mpc = self.add_cbf_constraints(mpc)
    
        mpc.setup()
        return mpc
    

    def add_cbf_constraints(self, mpc):
        """Adds the CBF constraints to the mpc controller. (MPC-CBF)

        Inputs:
          - mpc(do_mpc.controller.MPC): The mpc controller
        Returns:
          - mpc(do_mpc.controller.MPC): The mpc model with CBF constraints added
        """
        cbf_constraints = self.get_cbf_constraints()
        i = 0
        for cbc in cbf_constraints:
            mpc.set_nl_cons('cbf_constraint'+str(i), cbc, ub=0)
            i += 1
        return mpc

    def get_cbf_constraints(self):
        """Computes the CBF constraints for all obstacles.

        Returns:
          - cbf_constraints(list): The CBF constraints for each obstacle
        """
        # Get state vector x_{t+k+1}
        B = self.get_sys_matrix_B(self.model.x['x'])
        x_k1 = self.model.x['x'] + B@self.model.u['u']*self.Ts

        # Compute CBF constraints
        cbf_constraints = []
        if self.static_obstacles_on:
            for obs in self.obs:
                h_k1 = self.h(x_k1, obs)
                h_k = self.h(self.model.x['x'], obs)
                cbf_constraints.append(-h_k1 + (1-self.gamma)*h_k)

        if self.moving_obstacles_on:
            for i in range(len(self.moving_obs)):
                obs = (self.model.tvp['x_moving_obs'+str(i)], self.model.tvp['y_moving_obs'+str(i)], self.moving_obs[i][4])
                h_k1 = self.h(x_k1, obs)
                h_k = self.h(self.model.x['x'], obs)
                cbf_constraints.append(-h_k1 + (1-self.gamma)*h_k)

        return cbf_constraints

    def h(self, x, obstacle):
        """Computes the Control Barrier Function.
        
        Inputs:
          - x(casadi.casadi.SX): The state vector [3x1]
          - obstacle(tuple):     The obstacle position and radius
        Returns:
          - h(casadi.casadi.SX): The Control Barrier Function
        """
        x_obs, y_obs, r_obs = obstacle
        h = (x[0] - x_obs)**2 + (x[1] - y_obs)**2 - (self.r + r_obs + self.safety_dist)**2
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
            if self.control_type == "traj_tracking":
                # Trajectory to follow
                if config.trajectory == "circular":
                    x_traj = config.A*cos(config.w*t_now)
                    y_traj = config.A*sin(config.w*t_now)
                elif config.trajectory == "infinity":
                    x_traj = config.A*cos(config.w*t_now)/(sin(config.w*t_now)**2 + 1)
                    y_traj = config.A*sin(config.w*t_now)*cos(config.w*t_now)/(sin(config.w*t_now)**2 + 1)
                else:
                    print("Select one of the available options for trajectory.")
                    exit()

                tvp_struct_mpc['_tvp', :, 'x_set_point'] = x_traj
                tvp_struct_mpc['_tvp', :, 'y_set_point'] = y_traj

            if self.moving_obstacles_on is True:
                # Moving obstacles trajectory
                for i in range(len(self.moving_obs)):
                    tvp_struct_mpc['_tvp', :, 'x_moving_obs'+str(i)] = self.moving_obs[i][0]*t_now + self.moving_obs[i][1]
                    tvp_struct_mpc['_tvp', :, 'y_moving_obs'+str(i)] = self.moving_obs[i][2]*t_now + self.moving_obs[i][3]

            return tvp_struct_mpc

        mpc.set_tvp_fun(tvp_fun_mpc)
        return mpc

    def define_simulator(self):
        """Configures the simulator.

        Returns:
          - simulator(do_mpc.simulator.Simulator): The simulator
        """
        simulator = do_mpc.simulator.Simulator(self.model)
        simulator.set_param(t_step=self.Ts)

        # If trajectory tracking or moving obstacles: Add time-varying parameters
        if self.control_type == "traj_tracking" or self.moving_obstacles_on is True:
            tvp_template = simulator.get_tvp_template()

            def tvp_fun(t_now):
                return tvp_template
            simulator.set_tvp_fun(tvp_fun)

        simulator.setup()

        return simulator

    def set_init_state(self, x0):
        """Sets the initial state in all components."""
        self.mpc.setup()
        self.mpc.reset_history()
        self.simulator.reset_history()
        self.mpc.x0 = x0
        # print("Initial control:", self.mpc.u0['u'].T)
        self.mpc.u0 = np.zeros_like(self.mpc.u0['u'])
        # self.mpc.z0 = np.zeros_like(self.mpc.z0['z'])
        self.simulator.x0 = x0
        self.mpc.set_initial_guess()

    def run_simulation(self,x0):
        """Runs a closed-loop control simulation."""
        # print("Simualtor at beginning of simulation 1", self.simulator.x0)
        for k in range(self.sim_time):
            u0 = self.mpc.make_step(x0)
            x0 = self.simulator.make_step(u0)
            # y_next = self.simulator.make_step(u0, w0=10**(-4)*np.random.randn(3, 1))  # Optional Additive process noise
        # print("Simualtor at end of simulation 1", self.simulator.x0)
        return


    def run_simulation_to_get_final_condition(self,x0,xff,j,i):
        """Runs a closed-loop control simulation."""
        x1 = x0
        T=0.1
        epsilon=0.001
        # NOTE: For j < 3 the liveness value will be accurate lowkey.
        xf_minus_one=xff[j,0:2] 
        xf_one=xff[j-2,0:2]
        xf_minus_two=xff[j-1,0:2]
        xf_two=xff[j-3,0:2]
        # Will add liveliness condition here
        vec1=((xf_minus_two-xf_two)-(xf_minus_one-xf_one))/T#((xf_minus[j,0:2]-xf[j,0:2])-(xf_minus[i,0:2]-xf[i,0:2]))/T
        vec2=(xf_minus_two - xf_minus_one)#xf[j,0:2]-xf[i,0:2]
        l=np.arccos(abs(np.dot(vec1,vec2))/(LA.norm(vec1)*LA.norm(vec2)+epsilon))
        # print("X1 at beginning of final simulation", x1)
        # print("Simulator at beginning of final simulation", self.simulator.x0['x'])
        for k in range(self.sim_time):
            # print(f"\tSub-iteration")
            u1 = self.mpc.make_step(x1)
            # print(f"\tOutput of MPC: {u1.T}")
            u1_before_proj=u1
            if j>3 and i==1 and config.liveliness and l < config.liveness_threshold:
                # v = (xf_minus_two - xf_two)/T
                # norm_u1 = np.linalg.norm(u1_before_proj)
                # norm_v = np.linalg.norm(v)

                # # Special case: if u is the zero vector, return any point on the circle of radius half_norm_v   
                # if np.allclose(u1_before_proj, np.zeros_like(u1_before_proj)):
                #     # Example: [half_norm_v, 0]
                #     return np.array([norm_v / 2.5, 0])

                # # Scale u to have a norm half of that of v
                # u1 = (u1_before_proj / norm_u1) * (norm_u1 / 2)

                v0 = np.linalg.norm((xf_minus_two - xf_two)/T)
                v1 = np.linalg.norm((xf_minus_one - xf_one)/T)
                curr_v0_v1_point = np.array([v0, v1])
                desired_v0_v1_vec = np.array([1.0, 2.0])
                desired_v0_v1_point = np.dot(curr_v0_v1_point, desired_v0_v1_vec) / np.linalg.norm(desired_v0_v1_vec)
                control_vec = desired_v0_v1_point - curr_v0_v1_point
                control_vec_normalized = control_vec / np.linalg.norm(control_vec)
                control_mag = np.linalg.norm(u1_before_proj)
                u1 = control_vec_normalized * control_mag
                print(f"Running liveness {l}. Original control {u1_before_proj}. Control magnitude: {control_mag}. Output control {u1}")
                u1 = u1.reshape((2, 1))

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
