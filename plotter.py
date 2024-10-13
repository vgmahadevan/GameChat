import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # Create a figure and axis object for the plot
        self.liveliness_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    def init(self):
        # Reset plot limits and other properties as needed
        self.ax.set_xlim(-1, 2)
        self.ax.set_ylim(-1, 1)
        self.liveliness_text.set_text('Liveliness function = OFF')
        return []


    # Function to update the plots
    def update(self, frame):
        self.ax.clear()

        # Redraw static elements
        self.scenario.plot(self.ax)

        # Reset plot limits and other properties as needed
        self.ax.set_xlim(-2.6, 2.2)
        self.ax.set_ylim(-1.5, 1)

        # Determine the start index for the fading effect
        trail_length = 20
        start_index = max(0, frame - trail_length)  # Adjust '10' to control the length of the fading trail

        # Draw the fading trails for agent 1
        for i in range(start_index, frame):
            alpha = 1 - ((frame - i) / trail_length)**2
            self.ax.plot(self.x1[i:i+2, 0], self.x1[i:i+2, 1], 'r-', alpha=alpha, linewidth=5)

        # Draw the fading trails for agent 2
        for i in range(start_index, frame):
            alpha = 1 - ((frame - i) / trail_length)**2
            self.ax.plot(self.x2[i:i+2, 0], self.x2[i:i+2, 1], 'b-', alpha=alpha, linewidth=5)

        # Update the liveliness text
        # Your existing code to update liveliness text

        return []



    def plot(self, scenario, x1, x2, u, u_proj, L):
        self.scenario = scenario
        self.scenario.plot(self.ax)
        self.x1 = x1
        self.x2 = x2
        print("Plotting x1 and x2")
        print(self.x1)
        print(self.x2)

        # Draw the stationary circles
        # circles = [Circle((obstacles[i,0], obstacles[i,1]), obstacles[i,2], fill = False) for i in range(len(obstacles))]
        # circles = [Circle((obs[0], obs[1]), obs[2], fill = True) for obs in self.scenario.obstacles]

        # Create an animation
        ani = FuncAnimation(self.fig, lambda frame: self.update(frame), frames=len(self.x1), init_func=lambda: self.init(), blit=False)

        # Save the animation
        ani.save('agents_animation.mp4', writer='ffmpeg')
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


