import math
import config
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        # Create a figure and axis object for the plot
        self.liveliness_text = self.ax.text(0.05, 0.95, '', transform=self.ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    def init(self):
        # Reset plot limits and other properties as needed
        self.ax.set_xlim(-1, 2)
        self.ax.set_ylim(-2, 2)  # Adjusted y-axis limits
        #self.liveliness_text.set_text(f'Liveliness function = {config.liveliness}')
        return []


    # Function to update the plots
    def plot_live(self, scenario, controllers, x_cum, u_cum, scenario_type):
        self.scenario = scenario
        self.controllers = controllers
        self.x_cum = np.array(x_cum)
        self.u_cum = np.array(u_cum)
        self.update(len(x_cum[0]) - 1, scenario_type)
        plt.legend()
        plt.draw()
        plt.pause(0.01)
        if config.plot_live_pause:
            plt.waitforbuttonpress()


    # Function to update the plots
    def update(self, frame, scenario_type):
        self.ax.clear()

        frame *= config.plot_rate

        # Redraw static elements
        self.scenario.plot(self.ax)

        # Reset plot limits and other properties as needed
        self.ax.set_aspect('equal')
        if scenario_type == 0:
            self.ax.set_xlim(-1.5, 2.5)
            self.ax.set_ylim(-1.5, 1.5)  # Adjusted y-axis limits
        else:
            self.ax.set_xlim(-2.4, 1.2)
            self.ax.set_ylim(-2.4, 1.2)

        u0, u1 = np.round(self.u_cum[0][frame], 2), np.round(self.u_cum[1][frame], 2)
        try:
            L, u0_ori, u1_ori = np.round(self.controllers[0].liveliness[frame][0], 2), np.round(self.controllers[0].u_ori[frame], 2), np.round(self.controllers[1].u_ori[frame], 2)
        except Exception as e:
            print(e)
            L, u0_ori, u1_ori = 0, np.array([np.nan]), np.array([np.nan])
        x0_state, x1_state = self.x_cum[0][frame].T.copy(), self.x_cum[1][frame].T.copy()
        x0_state[2] = np.rad2deg(x0_state[2])
        x1_state = self.x_cum[1][frame].T.copy()
        x1_state[2] = np.rad2deg(x1_state[2])
        liveliness_text = [f'Liveliness function = {L}.',
                           f'Agent 0 X = {x0_state}.',
                           f'Agent 0 U_ori = {u0_ori.T}.',
                           f'Agent 0 U = {u0.T}.',
                           f'Agent 1 X = {x1_state}.',
                           f'Agent 1 U_ori = {u1_ori.T}.',
                           f'Agent 1 U = {u1.T}',
        ]
        #self.liveliness_text = self.ax.text(0.05, 0.95, '\n'.join(liveliness_text), transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Determine the start index for the fading effect
        trail_length = 20 * config.plot_rate
        start_index = max(0, frame - trail_length)  # Adjust '10' to control the length of the fading trail

        # Draw the fading trails for agents 1 and 2
        max_i = 0
        for i in range(start_index, frame - 1, config.plot_rate):
            max_i = max(max_i, i)
            alpha = max(0, min(.85, 1 - 1.5*((frame - 1 - i) / trail_length)**2))
            
        #     alpha = 1 if i == frame-2 else 0
        #    # Get the start and end points
        #     start_point = self.x_cum[0][i, :]
        #     end_point = self.x_cum[0][i+1, :]

        #     # Compute the direction vector from start to end point
        #     direction = end_point - start_point

        #     # Normalize the direction vector
        #     direction_norm = direction / np.linalg.norm(direction)

        #     # The second point is the center of the line, and the total length is 1
        #     # So the half-length is 0.5 units
        #     half_length = 0.5

        #     # The extended points: extend by half the total length in both directions
        #     extended_start = end_point - half_length * direction_norm
        #     extended_end = end_point + half_length * direction_norm

        #     line_length = np.linalg.norm(extended_end - extended_start)
        #     print(f"Line length: {line_length}")
        #     # Plot the extended line
        #     self.ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'r-', alpha=alpha, linewidth=5)



        #     # Get the start and end points
        #     start_point = self.x_cum[1][i, :]
        #     end_point = self.x_cum[1][i+1, :]

        #     # Compute the direction vector from start to end point
        #     direction = end_point - start_point

        #     # Normalize the direction vector
        #     direction_norm = direction / (np.linalg.norm(direction) + 0.001)

        #     # The second point is the center of the line, and the total length is 1
        #     # So the half-length is 0.5 units
        #     half_length = 0.5

        #     # The extended points: extend by half the total length in both directions
        #     extended_start = end_point - half_length * direction_norm
        #     extended_end = end_point + half_length * direction_norm

        #     # Plot the extended line
        #     self.ax.plot([extended_start[0], extended_end[0]], [extended_start[1], extended_end[1]], 'b-', alpha=alpha, linewidth=5)


            # self.ax.plot(self.x_cum[0][i:i+2, 0], self.x_cum[0][i:i+2, 1], 'r-', alpha=alpha, linewidth=5)
            # self.ax.plot(self.x_cum[1][i:i+2, 0], self.x_cum[1][i:i+2, 1], 'b-', alpha=alpha, linewidth=5)
            if scenario_type == 0:
                self.ax.plot(self.x_cum[0][i, 0], self.x_cum[0][i, 1], 'bo', alpha=alpha, markersize=9)
                self.ax.plot(self.x_cum[1][i, 0], self.x_cum[1][i, 1], 'ro', alpha=alpha, markersize=9)
            else:
                self.ax.plot(self.x_cum[0][i, 0], self.x_cum[0][i, 1], 'bo', alpha=alpha, markersize=8)
                self.ax.plot(self.x_cum[1][i, 0], self.x_cum[1][i, 1], 'ro', alpha=alpha, markersize=8)
        if scenario_type == 0:
            self.ax.plot(self.x_cum[0][max_i, 0], self.x_cum[0][max_i, 1], 'bo', alpha=1, markersize=16)
            self.ax.plot(self.x_cum[1][max_i, 0], self.x_cum[1][max_i, 1], 'ro', alpha=1, markersize=16)
        else:    
            self.ax.plot(self.x_cum[0][max_i, 0], self.x_cum[0][max_i, 1], 'bo', alpha=1, markersize=14)
            self.ax.plot(self.x_cum[1][max_i, 0], self.x_cum[1][max_i, 1], 'ro', alpha=1, markersize=14)
        
        pos_diff, vel_diff = self.controllers[0].liveliness[frame][1], self.controllers[0].liveliness[frame][2]
        # self.ax.arrow(0, 0, pos_diff[0], pos_diff[1], head_width=0.05, head_length=0.1, fc='green', ec='green', label='Position difference')
        # self.ax.arrow(0, 0, vel_diff[0], vel_diff[1], head_width=0.05, head_length=0.1, fc='orange', ec='orange', label='Velocity difference')
        plt.title("t = {:.2f} s".format(frame * config.Ts), fontsize=28)
        self.ax.tick_params(axis='both', labelsize=22)
        plt.savefig(f'plots/plot_{scenario_type}_{frame}.png')

        return []


    def plot(self, scenario, agents, x_cum, u_cum):
        self.scenario = scenario
        self.scenario.plot(self.ax)
        self.agents = agents
        self.x_cum = x_cum
        self.u_cum = u_cum

        # Create an animation
        ani = FuncAnimation(self.fig, lambda frame: self.update(frame), frames=len(self.x_cum[0]) // config.plot_rate, init_func=lambda: self.init(), blit=False)

        # Save the animation
        ani.save('agents_animation.mp4', writer='ffmpeg')

        # Set the color palette to "deep"
        sns.set_palette("deep")
        sns.set()
        fontsize = 14

        if config.dynamics == config.DynamicsModel.SINGLE_INTEGRATOR:
            speed1, speed2 = u_cum.copy()
            speed2_ori = agents[1].u_ori.copy()
            speed1 = [control[0] for control in speed1]
            speed2 = [control[0] for control in speed2]
            speed2_ori = [control[0] for control in speed2_ori]
        else:
            agent_1_states, agent_2_states = x_cum.copy()
            speed1 = [state[3] for state in agent_1_states]
            speed2 = [state[3] for state in agent_2_states]
            speed2_ori = speed2

        liveness = [l[0] for l in agents[0].liveliness.copy()]

        # Creating iteration indices for each agent based on the number of velocity points
        iterations = range(0, len(speed1), config.plot_rate)
        print("Iterations:", list(iterations))

        speed1 = [speed1[idx] for idx in iterations]
        speed2 = [speed2[idx] for idx in iterations]
        speed2_ori = [speed2_ori[idx] for idx in iterations]
        liveness = [liveness[idx] for idx in iterations]

        # Plotting the velocities as a function of the iteration for both agents
        plt.figure(figsize=(10, 10))

        # Plotting the x and y velocities for Agent 1
        plt.subplot(2, 1, 1)
        sns.lineplot(x=iterations, y=speed1, label='Agent 1 speed', marker='o', markers=True, dashes=False, markeredgewidth=0, linewidth=5, markersize=15)
        sns.lineplot(x=iterations, y=speed2, label='Agent 2 speed', marker='o', markers=True, dashes=False, markeredgewidth=0, linewidth=5, markersize=15)
        sns.lineplot(x=iterations, y=speed2_ori, label='Agent 2 speed original', marker='P', markers=True, dashes=False, markeredgewidth=0, linewidth=5, markersize=15)
        plt.xlabel('Iteration', fontsize=fontsize)
        plt.ylabel('Velocity', fontsize=fontsize)
        plt.legend(loc='upper left', ncol=1, fontsize=fontsize)
        plt.xlim(0, max(iterations))
        plt.ylim(min(speed1 + speed2 + speed2_ori), max(speed1 + speed2 + speed2_ori))
        plt.xticks(np.arange(0, max(iterations) + 1, 4 * config.plot_rate), fontsize=fontsize)
        plt.yticks(np.arange(round(min(speed1 + speed2 + speed2_ori), 1), round(max(speed1 + speed2 + speed2_ori), 1), .2), fontsize=fontsize)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
        plt.minorticks_on()

        plt.subplot(2, 1, 2)
        sns.lineplot(x=iterations, y=liveness, label='Liveness value', marker='o', color='orange', markers=True, dashes=False, markeredgewidth=0, linewidth=5, markersize=15)
        sns.lineplot(x=iterations, y=tuple(np.ones(len(iterations)) * config.liveness_threshold), label='Threshold', markers=True, dashes=False, markeredgewidth=0, linewidth=5, markersize=15)
        #plt.title('Liveness values', fontsize=fontsize)
        plt.xlabel('Iteration', fontsize=fontsize)
        plt.ylabel('Liveness', fontsize=fontsize)
        plt.legend(loc='upper left', ncol=1, fontsize=fontsize)
        plt.xlim(0, max(iterations))
        plt.ylim(0, 1.5)
        plt.xticks(np.arange(0, max(iterations) + 1, 4 * config.plot_rate), fontsize=fontsize)
        plt.yticks(np.arange(0, 1.6, .2), fontsize=fontsize)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
        plt.minorticks_on()

        plt.tight_layout()
        plt.show()


