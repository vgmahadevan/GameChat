"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
# x1 and x2 are time series states of agents 1 and 2 respectively
# n is the number of agents
# N is number of iterations for one time horizon
# mpc_cbf.py containst eh code for the Game thereotic MPC controllers for agent 1 and 2 respectively

import config
import numpy as np
from mpc_cbf import MPC
from scenarios import DoorwayScenario, IntersectionScenario
from plotter import Plotter
from data_logger import DataLogger

# Add number of agents
n=2
N=int(20.0 / config.Ts) # Number of iteration steps for each agent
# N = 10

xf=np.zeros((n,config.num_states)) # initialization of final states
u = []
L=[]
u_proj = []

# final_positions_both_agents stores the final positions of both agents
# [1,:], [3,:], [5,:]... are final positions of agent 1
# [2,:], [4,:], [6,:]... are final positions of agent 2
final_positions_both_agents=np.zeros((n*N,config.num_states)) # initialization of final states for both agents

x1=np.zeros((N,config.num_states)) # initialization of times series states of agent 1 
x2=np.zeros((N,config.num_states)) # initialization of times series states of agent 1 

x_cum = [[], []]

xf_minus=np.zeros((n,config.num_states))

# Scenarios: "doorway" or "intersection"
scenario = DoorwayScenario()
# scenario = IntersectionScenario()

# Matplotlib plotting handler
plotter = Plotter()
logger = DataLogger('doorway_train_data_no_liveness.json')

def main():
    c=0
    # Add all initial and goal positions of the agents here (Format: [x, y, theta])
    initial = scenario.initial.copy()
    goals = scenario.goals.copy()
    logger.set_obstacles(scenario.obstacles.copy())
    for j in range(N): # j denotes the j^th step in one time horizon
        for i in range(n):  # i is the i^th agent 
            # Define controller & run simulation for each agent i
            obs = scenario.obstacles.copy()

            # Ensures that other agents act as obstacles to agent i
            for k in range(n):
                if i != k:
                    obs.append((initial[k,0], initial[k,1], config.agent_radius)) 

            # Initialization of MPC controller for the ith agent
            # The position of agent i is propogated one time horizon ahead using the MPC controller
            print(f"\n\nIteration {j}, Agent {i}")
            final_positions_both_agents[c,:] = xf[i,:]

            controller = MPC(agent_idx=i, initial_state=initial[i,:], goal=goals[i,:], static_obs=obs, opp_state=initial[1-i,:])
            controller.set_init_state(initial[i,:])
            # controller.run_simulation(initial[i,:])
            x, uf, uf_proj, l = controller.run_simulation_to_get_final_condition(initial[i,:],final_positions_both_agents,j)
            # opp_state = (initial[1-i,0], initial[1-i,1])
            # data_input = np.concatenate((x, opp_state), axis=None)
            # logger.log_iteration(data_input, uf)

            xf[i,:] = x.ravel()
            x_cum[i].append(x.ravel())
            u.append(uf)
            u_proj.append(uf_proj)
            L.append(l)

            print(f"Initial state: {initial[i, :]}, Output control: {uf_proj}, New state: {xf[i, :]}")

            c += 1
   
        # Plots
        initial = xf.copy() #The final state is assigned to the initial state stack for future MPC
        if j % config.plot_rate == 0:
            plotter.plot_live(scenario, x_cum, u, u_proj, L)

    #x1 and x2 are times series data of positions of agents 1 and 2 respectively
    for ll in range(N-1):
        x1[ll,:]=final_positions_both_agents[n*ll,:]
        x2[ll,:]=final_positions_both_agents[n*ll+1,:]
    
    # Discard the first element of both x1 and x2
    plotter.plot(scenario, x1[1:], x2[1:], u, u_proj, L)
       


if __name__ == '__main__':
    main()
