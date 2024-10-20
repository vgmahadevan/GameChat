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
        self.ax.set_ylim(-1, 1)
        self.liveliness_text.set_text(f'Liveliness function = {config.liveliness}')
        return []


    # Function to update the plots
    def plot_live(self, scenario, x_cum, u, u_proj, L):
        self.ax.clear()
        self.ax.set_xlim(-1, 2)
        self.ax.set_ylim(-1, 1)

        x1, x2 = x_cum
        
        # Draw the fading trails for agent 1
        L, u, u_proj = np.round(L, 2), np.round(u, 2), np.round(u_proj, 2)
        x0_state = x1[-1].T.copy()
        x0_state[2] = np.rad2deg(x0_state[2])
        x1_state = x2[-1].T.copy()
        x1_state[2] = np.rad2deg(x1_state[2])
        liveliness_text = [f'Liveliness function = {L[-1]}.',
                           f'Agent 0 X = {x0_state}.',
                           f'Agent 0 U = {u[-2].T}.',
                           f'Agent 0 U_proj = {u_proj[-2].T}.',
                           f'Agent 1 X = {x1_state}.',
                           f'Agent 1 U = {u[-1].T}.',
                           f'Agent 1 U_proj = {u_proj[-1].T}',
        ]
        self.liveliness_text = self.ax.text(0.05, 0.95, '\n'.join(liveliness_text), transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Redraw static elements
        scenario.plot(self.ax)

        # Reset plot limits and other properties as needed
        self.ax.set_xlim(-2.6, 2.2)
        self.ax.set_ylim(-1, 1)
 
        # Determine the start index for the fading effect
        frame = len(x1)
        trail_length = 20 * config.plot_rate
        start_index = max(0, frame - trail_length)  # Adjust '10' to control the length of the fading trail

        # Draw the fading trails for agent 1
        for i in range(start_index, frame, config.plot_rate):
            alpha = 1 - ((frame - 1 - i) / trail_length)**2
            self.ax.scatter(x1[i][0], x1[i][1], c='r', s=25, alpha=alpha)
            self.ax.scatter(x2[i][0], x2[i][1], c='b', s=25, alpha=alpha)

        if frame > 2:
            dp = x1[-1][:2] - x2[-1][:2]
            v1 = x1[-1][:2] - x1[-2][:2]
            v2 = x2[-1][:2] - x2[-2][:2]
            dv = v1 - v2
            
            plt.arrow(0, 0, dp[0], dp[1], color='m', linewidth=2)
            plt.arrow(0, 0, dv[0], dv[1], color='g', linewidth=2)


        # Update the liveliness text
        # Your existing code to update liveliness text
        plt.draw()
        plt.pause(0.01)
        if config.plot_live_pause:
            plt.waitforbuttonpress()


    # Function to update the plots
    def update(self, frame):
        self.ax.clear()

        frame *= config.plot_rate

        # Redraw static elements
        self.scenario.plot(self.ax)

        # Reset plot limits and other properties as needed
        self.ax.set_xlim(-2.6, 2.2)
        self.ax.set_ylim(-1, 1)

        L, u0, u0_proj, u1, u1_proj = np.round(self.L[frame], 2), np.round(self.u[frame*2], 2), np.round(self.u_proj[frame*2], 2), np.round(self.u[frame*2+1], 2), np.round(self.u_proj[frame*2+1], 2)
        x0_state = self.x1[frame].T.copy()
        x0_state[2] = np.rad2deg(x0_state[2])
        x1_state = self.x2[frame].T.copy()
        x1_state[2] = np.rad2deg(x1_state[2])
        liveliness_text = [f'Liveliness function = {L}.',
                           f'Agent 0 X = {x0_state}.',
                           f'Agent 0 U = {u0.T}.',
                           f'Agent 0 U_proj = {u0_proj.T}.',
                           f'Agent 1 X = {x1_state}.',
                           f'Agent 1 U = {u1.T}.',
                           f'Agent 1 U_proj = {u1_proj.T}',
        ]
        self.liveliness_text = self.ax.text(0.05, 0.95, '\n'.join(liveliness_text), transform=self.ax.transAxes, fontsize=10, verticalalignment='top')

        # Determine the start index for the fading effect
        trail_length = 20 * config.plot_rate
        start_index = max(0, frame - trail_length)  # Adjust '10' to control the length of the fading trail

        # Draw the fading trails for agent 1
        for i in range(start_index, frame, config.plot_rate):
            alpha = 1 - ((frame - i) / trail_length)**2
            self.ax.plot(self.x1[i:i+2, 0], self.x1[i:i+2, 1], 'r-', alpha=alpha, linewidth=5)

        # Draw the fading trails for agent 2
        for i in range(start_index, frame, config.plot_rate):
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
        self.u = u
        self.u_proj = u_proj
        self.L = L

        # Create an animation
        ani = FuncAnimation(self.fig, lambda frame: self.update(frame), frames=len(self.x1) // config.plot_rate, init_func=lambda: self.init(), blit=False)

        # Save the animation
        ani.save('agents_animation.mp4', writer='ffmpeg')
        # Adjusting the font to "Segoe UI"
        # plt.rcParams['font.size'] = 200

        # Set the color palette to "deep"
        sns.set_palette("deep")
        sns.set()
        fontsize = 14

        # TODO: Fix this.
        agent_1_velocities = [v.ravel() for i, v in enumerate(u) if i % 2 == 0] 
        agent_2_velocities = [v.ravel() for i, v in enumerate(u) if i % 2 != 0]
        agent_2_velocities_proj = [v.ravel() for i, v in enumerate(u_proj) if i % 2 != 0]
        liveness = [v for i, v in enumerate(L) if i % 2 != 0]

        # Creating iteration indices for each agent based on the number of velocity points
        iterations = range(0, len(agent_2_velocities), config.plot_rate)
        print("Iterations:", list(iterations))

        # Unpacking the velocities into x and y components for both agents
        agent_1_x, agent_1_y = zip(*agent_1_velocities)
        speed1 = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_1_x, agent_1_y))
        agent_2_x, agent_2_y = zip(*agent_2_velocities)
        speed2 = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_2_x, agent_2_y))
        agent_2_x_proj, agent_2_y_proj = zip(*agent_2_velocities_proj)
        speed2_proj = tuple(math.sqrt(x**2 + y**2) for x, y in zip(agent_2_x_proj, agent_2_y_proj))

        speed1 = [speed1[idx] for idx in iterations]
        speed2 = [speed2[idx] for idx in iterations]
        speed2_proj = [speed2_proj[idx] for idx in iterations]
        liveness = [liveness[idx] for idx in iterations]

        # Plotting the velocities as a function of the iteration for both agents
        plt.figure(figsize=(10, 10))

        # Plotting the x and y velocities for Agent 1
        plt.subplot(2, 1, 1)
        sns.lineplot(x=iterations, y=speed1, label='Agent 1 speed', marker='o',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
        sns.lineplot(x=iterations, y=speed2, label='Agent 2 speed', marker='o',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
        sns.lineplot(x=iterations, y=speed2_proj, label='Agent 2 speed projected', marker='P',markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
        plt.xlabel('Iteration', fontsize = fontsize)
        plt.ylabel('Velocity', fontsize = fontsize)
        plt.legend(loc='upper left', ncol=1, fontsize = fontsize)
        plt.xlim(0, max(iterations))
        plt.ylim(min(speed1 + speed2+ speed2_proj), max(speed1 + speed2+ speed2_proj))
        plt.xticks(np.arange(0, max(iterations)+1, 4*config.plot_rate), fontsize = fontsize)
        plt.yticks(np.arange(round(min(speed1 + speed2+ speed2_proj), 1), round(max(speed1 + speed2+ speed2_proj), 1), .2), fontsize = fontsize)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
        plt.minorticks_on()

        plt.subplot(2, 1, 2)
        sns.lineplot(x=iterations, y=liveness, label='Liveness value', marker='o', color = 'orange', markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
        sns.lineplot(x=iterations, y=tuple(np.ones(len(iterations))*.3), label='Threshold', markers=True, dashes=False,markeredgewidth=0, linewidth = 5, markersize = 15)
        #plt.title('Liveness values', fontsize = fontsize)
        plt.xlabel('Iteration', fontsize = fontsize)
        plt.ylabel('Liveness', fontsize = fontsize)
        plt.legend(loc='upper left', ncol=1, fontsize = fontsize)
        plt.xlim(0, max(iterations))
        plt.ylim(0, 1.5)
        plt.xticks(np.arange(0, max(iterations)+1, 4*config.plot_rate), fontsize = fontsize)
        plt.yticks(np.arange(0, 1.6, .2), fontsize = fontsize)
        plt.grid(which='major', color='#CCCCCC', linestyle='--')
        plt.grid(which='minor', color='#CCCCCC', linestyle='-', axis='both')
        plt.minorticks_on()

        plt.tight_layout()
        plt.show()
        plt.show()


