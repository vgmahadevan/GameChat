import do_mpc
from casadi import *
import config
from config import DynamicsModel
from numpy import linalg as LA
import numpy as np

EPSILON = 0.001

class System:
    def __init__(self, agent_idx, initial_state):
        self.agent_idx = agent_idx
        self.initial_state = initial_state

        self.model = self.define_model()
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
        A, B = self.get_dynamics(_x)

        # Set right-hand-side of ODE for all introduced states (_x).
        x_next = _x + A*config.Ts + B@_u*config.Ts
        model.set_rhs('x', x_next, process_noise=False)  # Set to True if adding noise

        # Setup model
        model.setup()
        return model

    @staticmethod
    def get_dynamics(x):
        if config.dynamics == DynamicsModel.SINGLE_INTEGRATOR:
            return System.get_single_integrator_dynamics(_x)
        else:
            return System.get_double_integrator_dynamics(_x)


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

    def run_simulation_to_get_final_condition(self, controller):
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
