"""Game theoretic MPC-CBF controller for a differential drive mobile robot."""

# State: [x, y, theta], Position: [x, y]
#x1 and x2 are time series states of agents 1 and 2 respectively
#n is the number of agents
#N is number of iterations for one time horizon
#controller[1] and controller[2] are the Game thereotic MPC controllers for agent 1 and 2 respectively

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpc_cbf import MPC
from plotter import Plotter
import util
import config
from numpy import linalg as LA
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import casadi as ca

# Add number of agents
n=2
N=50 #Number of iteration steps for each agent

controller=[None]*n #Control input intitialization
xf=np.zeros((n,3)) # initialization of final states
u = []
L=[]
u_proj = []
final_positions_both_agents=np.zeros((n*N,3)) # initialization of final states for both agents


x1=np.zeros((N,3)) # initialization of times series states of agent 1 
x2=np.zeros((N,3)) # initialization of times series states of agent 1 

xf_minus=np.zeros((n,3))
def main():
    c=0
    Q=config.Q_sp
    R=config.R_sp

    #Scenarios: "doorway" or "intersection"
    scenario="doorway"
    #Add all initial and goal positions of the agents here (Format: [x, y, theta])
    if scenario=="doorway":
        initial=np.array([[-1, 0.5, 0],
                    [-1, -0.5, 0]])
        # initial=np.array([[-1, 0.5, 0],
        #             [-2.5, -1, 0]])
        # initial=np.array([[-2.5, 0.5, 0],
        #             [-2.5, -0.5, 0]])
        goals=np.array([[2, 0, 0],
                    [2, 0, 0]])
        # goals=np.array([[2, 0.1, 0],
        #             [2, -0.1, 0]])
        
    else:
        initial=np.array([[0.0, -2.0, 0],
                      [-2.0, 0.0, 0]])
        goals=np.array([[0.0, 1.0, 0],
                    [1.0, 0.0, 0]
                    ])

    if scenario=="doorway":
        ox=1
    else:
        ox=-0.3
        ox1=0.3
    LL=[]
    first_time = True
    for j in range(N): # j denotes the j^th step in one time horizon
        for i in range(n):  # i is the i^th agent 

        # Define controller & run simulation for each agent i
            config.x0=initial[i,:]
            config.goal=goals[i,:]
            config.obs=[]
            if scenario=="doorway":
                # config.obs= [(ox, 0.4, 0.1),(ox, 0.5, 0.1),(ox, 0.6, 0.1),(ox, 0.7, 0.1),(ox, 0.8, 0.1),(ox, 0.9, 0.1),
                #   (ox, 1.0, 0.1),(ox, -0.3, 0.1), (ox, -0.3, 0.1),(ox, -0.3, 0.1),(ox, -0.4, 0.1),(ox, -0.5, 0.1),(ox, -0.6, 0.1),(ox, -0.7, 0.1),      
                # (ox, -0.8, 0.1),(ox, -0.9, 0.1)]
                config.obs=[(ox, 0.3, 0.1),(ox, 0.4, 0.1),(ox, 0.5, 0.1),(ox, 0.6, 0.1),(ox, 0.7, 0.1),(ox, 0.8, 0.1),(ox, 0.9, 0.1), (ox, 1.0, 0.1), (ox, -0.3, 0.1),(ox, -0.4, 0.1),(ox, -0.5, 0.1),(ox, -0.6, 0.1),(ox, -0.7, 0.1), (ox, -0.8, 0.1),(ox, -0.9, 0.1),(ox, -1.0, 0.1)]
    

                obstacles=config.obs
                

            else:
                config.obs= [(ox, 0.3, 0.1),(ox, 0.4, 0.1),(ox, 0.5, 0.1),(ox, 0.6, 0.1),(ox, 0.7, 0.1),(ox, 0.8, 0.1),(ox, 0.9, 0.1),
                  (ox, 1.0, 0.1), (ox, -0.3, 0.1),(ox, -0.4, 0.1),(ox, -0.5, 0.1),(ox, -0.6, 0.1),(ox, -0.7, 0.1),      
                (ox, -0.8, 0.1),(ox, -0.9, 0.1),(ox, -1.0, 0.1),

                (ox1, 0.3, 0.1),(ox1, 0.4, 0.1),(ox1, 0.5, 0.1),(ox1, 0.6, 0.1),(ox1, 0.7, 0.1),(ox1, 0.8, 0.1),(ox1, 0.9, 0.1),(ox1, 1.0, 0.1),
                  (ox1, -0.3, 0.1),(ox1, -0.4, 0.1),(ox1, -0.5, 0.1),(ox, -0.6, 0.1),(ox1, -0.7, 0.1),(ox1, -0.8, 0.1),(ox1, -0.9, 0.1),(ox1, -1.0, 0.1),
                  
                  (0.3,ox, 0.1), (0.4,ox, 0.1),( 0.5,ox, 0.1),(0.6,ox, 0.1),( 0.7,ox, 0.1),(0.8,ox, 0.1),(0.9,ox, 0.1),(1.0,ox, 0.1),
                  (-0.3,ox, 0.1), (-0.4,ox, 0.1),(-0.5,ox, 0.1),(-0.6,ox, 0.1),(-0.7,ox, 0.1),(-0.8,ox, 0.1),(-0.9,ox, 0.1),(-1.0,ox, 0.1),

                (0.3,ox1, 0.1), ( 0.4,ox1, 0.1),(0.5,ox1, 0.1),(0.6, ox1, 0.1),(0.7,ox1, 0.1),(0.8,ox1, 0.1),(0.9,ox1, 0.1),( 1.0, ox1, 0.1),
                (-0.3,ox1, 0.1), ( -0.4,ox1, 0.1),(-0.5, ox1, 0.1),( -0.6, ox1, 0.1),( -0.7,ox1, 0.1),( -0.8,ox1, 0.1),( -0.9,ox1, 0.1),(-1.0,ox1, 0.1)]
                obstacles=config.obs


            # Setting the maximum velocity limits for both agents
            if i==0:
                config.v_limit=0.3 #For agent 1
            else:
                # config.v_limit=0.6    #For agent 2
                config.v_limit=0.3  #For agent 2

            # Ensures that other agents act as obstacles to agent i
            for k in range(n):
                if i!=k:
                    config.obs.append((initial[k,0], initial[k,1],0.08)) 

            #Liveliness "on" or "off" can be selected from here
            liveliness='on'

            #Initialization of MPC controller for the ith agent
            controller[i] = MPC(final_positions_both_agents,j,i,liveliness)

            #final_positions_both_agents stores the final positions of both agents
            #[1,:], [3,:], [5,:]... are final positions of agent 1
            #[2,:], [4,:], [6,:]... are final positions of agent 2
            final_positions_both_agents[c,:]=xf[i,:]
            lll=controller[i].run_simulation(final_positions_both_agents,j)
            LL.append(lll)
            #The position of agent i is propogated one time horizon ahead using the MPC controller
            x, uf, uf_proj, l =controller[i].run_simulation_to_get_final_condition(final_positions_both_agents,j,i,liveliness, first_time)
            xf[i,:] = x.ravel()            
            u.append(uf)
            u_proj.append(uf_proj)
            L.append(l)

            c=c+1
   
        # Plots
        initial=xf #The final state is assigned to the initial state stack for future MPC

    #x1 and x2 are times series data of positions of agents 1 and 2 respectively
    for ll in range(N-1):
        x1[ll,:]=final_positions_both_agents[n*ll,:]
        x2[ll,:]=final_positions_both_agents[n*ll+1,:]
    
    # Discard the first element of both x1 and x2
    x1_ = x1[1:, :]
    x2_ = x2[1:, :]
    # T=0.4
    # L=[]
    # # Computation of liveliness value L for agent 1
    # for i in range(len(x1)-2):
    #     vec1=((x2[i+1,0:2]-x2[i,0:2])-(x1[i+1,0:2]-x1[i,0:2]))/T
    #     vec2=x1[i,0:2]-x2[i,0:2]
    #     l=np.arccos(np.dot(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2)+0.001))
    #     l_sin=abs(np.arcsin(np.cross(vec1,vec2)/(LA.norm(vec1)*LA.norm(vec2)+0.001)))
    #     print(l, l_sin)
    #     L.append(l)

    #Everything below is plotting

    # Create a figure and axis object for the plot
    fig, ax = plt.subplots()
    liveliness_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Draw the stationary circles
    # circles = [Circle((obstacles[i,0], obstacles[i,1]), obstacles[i,2], fill = False) for i in range(len(obstacles))]
    circles = [Circle((obs[0], obs[1]), obs[2], fill = True) for obs in obstacles]
    if scenario=="doorway":
        rect = patches.Rectangle((ox-0.1,0.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect1 = patches.Rectangle((ox-0.1,-1.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        ax.add_patch(rect)
        ax.add_patch(rect1)
    else:
        length=1
        rect = patches.Rectangle((ox-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect1 = patches.Rectangle((ox1-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect2 = patches.Rectangle((ox1-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect3 = patches.Rectangle((ox-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect4 = patches.Rectangle((-length,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect5 = patches.Rectangle((-length,ox1-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect6 = patches.Rectangle((ox1-0.1,ox1-0.1),1-ox1+0.2,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect7 = patches.Rectangle((ox1,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        ax.add_patch(rect)
        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)
        ax.add_patch(rect4)
        ax.add_patch(rect5)
        ax.add_patch(rect6)
        ax.add_patch(rect7)

    # for circle in circles:
    #     ax.add_patch(circle) 
       
# Initialize two empty plots for agent 1 and agent 2
    # agent1, = plt.plot([], [], 'ro', markersize=5)
    # agent2, = plt.plot([], [], 'bo', markersize=5)
    agent1_line, = plt.plot([], [], 'r-', linewidth=2)
    agent2_line, = plt.plot([], [], 'b-', linewidth=2)
# Function to initialize the plots, returns the plot objects
    # def init():   
    #     ax.set_xlim(-2, 4)
    #     ax.set_ylim(-2, 4)
    #     liveliness_text.set_text('Liveliness function = OFF')

    #     return agent1, agent2,liveliness_text,

    def init():
        # Reset plot limits and other properties as needed
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 1)
        liveliness_text.set_text('Liveliness function = OFF')
        return []
    
    def redraw_static_elements():
        if scenario=="doorway":
            rect = patches.Rectangle((ox-0.1,0.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect1 = patches.Rectangle((ox-0.1,-1.3),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            ax.add_patch(rect)
            ax.add_patch(rect1)
        else:
            length=1
            rect = patches.Rectangle((ox-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect1 = patches.Rectangle((ox1-0.1,-length),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect2 = patches.Rectangle((ox1-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect3 = patches.Rectangle((ox-0.1,ox1),0.2,1-ox1+0.1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect4 = patches.Rectangle((-length,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect5 = patches.Rectangle((-length,ox1-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect6 = patches.Rectangle((ox1-0.1,ox1-0.1),1-ox1+0.2,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            rect7 = patches.Rectangle((ox1,ox-0.1),1-ox1+0.1,0.2,linewidth=1,edgecolor='k',facecolor='k',fill=True)
            ax.add_patch(rect)
            ax.add_patch(rect1)
            ax.add_patch(rect2)
            ax.add_patch(rect3)
            ax.add_patch(rect4)
            ax.add_patch(rect5)
            ax.add_patch(rect6)
            ax.add_patch(rect7)
    # Function to update the plots
    def update(frame):
        ax.clear()

        # Redraw static elements
        redraw_static_elements()

        # Reset plot limits and other properties as needed
        ax.set_xlim(-2.6, 2.2)
        ax.set_ylim(-1.5, 1)
        # ax.set_title("Your Title Here")
        # ax.set_xlabel("X Axis Label")
        # ax.set_ylabel("Y Axis Label")

        # Determine the start index for the fading effect
        trail_length = 20
        start_index = max(0, frame - trail_length)  # Adjust '10' to control the length of the fading trail

        # Draw the fading trails for agent 1
        for i in range(start_index, frame):
            alpha = 1 - ((frame - i) / trail_length)**2
            ax.plot(x1_[i:i+2, 0], x1_[i:i+2, 1], 'r-', alpha=alpha, linewidth=5)

        # Draw the fading trails for agent 2
        for i in range(start_index, frame):
            alpha = 1 - ((frame - i) / trail_length)**2
            ax.plot(x2_[i:i+2, 0], x2_[i:i+2, 1], 'b-', alpha=alpha, linewidth=5)

        # Update the liveliness text
        # Your existing code to update liveliness text

        return []
    # Create an animation
    ani = FuncAnimation(fig, update, frames=len(x1), init_func=init, blit=False)

    # Save the animation
    ani.save('agents_animation.mp4', writer='ffmpeg')
    import seaborn as sns
    import math
    # Adjusting the font to "Segoe UI"
    plt.rcParams['font.size'] = 200

    # Set the color palette to "deep"
    sns.set_palette("deep")
    sns.set()
    fontsize = 36
    agent_1_velocities = [v.ravel() for i, v in enumerate(u[:40]) if i % 2 == 0] 
    agent_2_velocities = [v.ravel() for i, v in enumerate(u[:40]) if i % 2 != 0]
    agent_2_velocities_proj = [v.ravel() for i, v in enumerate(u_proj[:40]) if i % 2 != 0]
    liveness = [v for i, v in enumerate(L[:40]) if i % 2 != 0]

    # Unpacking the velocities into x and y components for both agents
    agent_1_x, agent_1_y = zip(*agent_1_velocities)
    speed1 = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_1_x, agent_1_y))
    agent_2_x, agent_2_y = zip(*agent_2_velocities)
    speed2 = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_2_x, agent_2_y))
    agent_2_x_proj, agent_2_y_proj = zip(*agent_2_velocities_proj)
    speed2_proj = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_2_x_proj, agent_2_y_proj))

    # Creating iteration indices for each agent based on the number of velocity points
    iterations = range(len(agent_2_velocities))

    # Plotting the velocities as a function of the iteration for both agents
    plt.figure(figsize=(25, 14))

    # Plotting the x and y velocities for Agent 1
    plt.subplot(2, 1, 1)
    sns.lineplot(x=iterations, y=speed1, label='Agent 1 speed', marker='o',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
    sns.lineplot(x=iterations, y=speed2, label='Agent 2 speed', marker='o',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
    sns.lineplot(x=iterations, y=speed2_proj, label='Agent 2 speed projected', marker='P',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
    # sns.lineplot(x=iterations, y=agent_2_x_proj, label='Agent 2 Proj - X Velocity', marker='o',markeredgewidth=0)
    # sns.lineplot(x=iterations, y=agent_2_y_proj, label='Agent 2 Proj - Y Velocity', marker='P',markeredgewidth=0)
    #plt.title('Agent Velocities', fontsize = fontsize)
    plt.xlabel('Iteration', fontsize = fontsize)
    plt.ylabel('Velocity', fontsize = fontsize)
    plt.legend(loc='upper left', ncol=1, fontsize = fontsize)
    plt.xlim(0, max(iterations))
    plt.ylim(min(speed1 + speed2+ speed2_proj), max(speed1 + speed2+ speed2_proj))
    plt.xticks(np.arange(0, max(iterations)+1, 4), fontsize = fontsize)
    plt.yticks(np.arange(round(min(speed1 + speed2+ speed2_proj), 1), round(max(speed1 + speed2+ speed2_proj), 1), .2), fontsize = fontsize)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
    plt.minorticks_on()


    # plt.subplot(3, 1, 2)
    # sns.lineplot(x=iterations, y=agent_2_x, label='Agent 2 - X Velocity', marker='o',markers=True, dashes=False,markeredgewidth=0)
    # sns.lineplot(x=iterations, y=agent_2_y, label='Agent 2 - Y Velocity', marker='P',markers=True, dashes=False,markeredgewidth=0)
    # plt.title('Agent 2 Velocities Before Projection')
    # plt.xlabel('Iteration')
    # plt.ylabel('Velocity')
    # plt.legend(loc='lower left', ncol=2)    
    # plt.xlim(0, max(iterations))
    # plt.ylim(min(agent_2_x + agent_2_y), max(agent_2_x + agent_2_y))
    # plt.xticks(np.arange(0, max(iterations)+1, 4))
    # plt.yticks(np.arange(round(min(agent_2_x + agent_2_y), 1), round(max(agent_2_x + agent_2_y), 1), .5))
    # plt.grid(which='major', color='#CCCCCC', linestyle='--')
    # plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
    # plt.minorticks_on()


    plt.subplot(2, 1, 2)
    sns.lineplot(x=iterations, y=liveness, label='Liveness value', marker='o', color = 'orange', markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
    sns.lineplot(x=iterations, y=tuple(np.ones(len(iterations))*.3), label='Threshold', markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
    #plt.title('Liveness values', fontsize = fontsize)
    plt.xlabel('Iteration', fontsize = fontsize)
    plt.ylabel('Liveness', fontsize = fontsize)
    plt.legend(loc='upper left', ncol=1, fontsize = fontsize)
    plt.xlim(0, max(iterations))
    plt.ylim(0, 1.5)
    plt.xticks(np.arange(0, max(iterations)+1, 4), fontsize = fontsize)
    plt.yticks(np.arange(0, 1.6, .2), fontsize = fontsize)
    plt.grid(which='major', color='#CCCCCC', linestyle='--')
    plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
    plt.minorticks_on()

    plt.tight_layout()
    plt.show()
    plt.show()




if __name__ == '__main__':
    main()
    # Extracting the velocities for agent 1 and agent 2

