import config
from config import DynamicsModel
import numpy as np
import matplotlib.patches as patches

#Add all initial and goal positions of the agents here (Format: [x, y, theta])

class DoorwayScenario:
    def __init__(self):
        self.num_agents = 2
        goal_y = 0.2
        self.initial = np.array([[-1.05, 0.5, -np.radians(10)],
                    [-1, -0.5, np.radians(10)]])
        self.goals = np.array([[2, -goal_y, 0.0],
                    [2, goal_y, 0.0]])
        if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
            # Set initial state to 0 velocity and goal to 0 velocity.
            zeros = np.zeros((self.num_agents, 1))
            self.initial = np.hstack((self.initial, zeros))
            self.goals = np.hstack((self.goals, zeros))
        self.ox=1
        self.ox_size = 0.05
        self.obstacles=[(self.ox, 0.25, self.ox_size), (self.ox, 0.3, self.ox_size),(self.ox, 0.4, self.ox_size),(self.ox, 0.5, self.ox_size),(self.ox, 0.6, self.ox_size),(self.ox, 0.7, self.ox_size),(self.ox, 0.8, self.ox_size),(self.ox, 0.9, self.ox_size), (self.ox, 1.0, self.ox_size), (self.ox, 0.25, self.ox_size), (self.ox, -0.3, self.ox_size),(self.ox, -0.4, self.ox_size),(self.ox, -0.5, self.ox_size),(self.ox, -0.6, self.ox_size),(self.ox, -0.7, self.ox_size), (self.ox, -0.8, self.ox_size),(self.ox, -0.9, self.ox_size),(self.ox, -1.0, self.ox_size)]
    
    def plot(self, ax):
        rect = patches.Rectangle((self.ox-0.1,0.25),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        rect1 = patches.Rectangle((self.ox-0.1,-1.25),0.2,1,linewidth=1,edgecolor='k',facecolor='k',fill=True)
        ax.add_patch(rect)
        ax.add_patch(rect1)
        pass


class NoObstacleDoorwayScenario:
    def __init__(self):
        self.num_agents = 2
        goal_y = 0.2
        self.initial = np.array([[-1, 0.5, 0],
                    [-1, -0.5, 0]])
        self.goals = np.array([[2, -goal_y, 0.0],
                    [2, goal_y, 0.0]])
        if config.dynamics == DynamicsModel.DOUBLE_INTEGRATOR:
            # Set initial state to 0 velocity and goal to 0 velocity.
            zeros = np.zeros((self.num_agents, 1))
            self.initial = np.hstack((self.initial, zeros))
            self.goals = np.hstack((self.goals, zeros))
        self.obstacles = []
    
    def plot(self, ax):
        pass


class IntersectionScenario:
    def __init__(self):
        self.initial = np.array([[0.0, -2.0, 0],
                      [-2.0, 0.0, 0]])
        self.goals = np.array([[0.0, 1.0, 0],
                    [1.0, 0.0, 0]
                    ])
        self.ox=-0.3
        self.ox1=0.3

        self.obstacles=[(self.ox, 0.3, 0.1),(self.ox, 0.4, 0.1),(self.ox, 0.5, 0.1),(self.ox, 0.6, 0.1),(self.ox, 0.7, 0.1),(self.ox, 0.8, 0.1),(self.ox, 0.9, 0.1),
                  (self.ox, 1.0, 0.1), (self.ox, -0.3, 0.1),(self.ox, -0.4, 0.1),(self.ox, -0.5, 0.1),(self.ox, -0.6, 0.1),(self.ox, -0.7, 0.1),      
                (self.ox, -0.8, 0.1),(self.ox, -0.9, 0.1),(self.ox, -1.0, 0.1),

                (self.ox1, 0.3, 0.1),(self.ox1, 0.4, 0.1),(self.ox1, 0.5, 0.1),(self.ox1, 0.6, 0.1),(self.ox1, 0.7, 0.1),(self.ox1, 0.8, 0.1),(self.ox1, 0.9, 0.1),(self.ox1, 1.0, 0.1),
                  (self.ox1, -0.3, 0.1),(self.ox1, -0.4, 0.1),(self.ox1, -0.5, 0.1),(self.ox, -0.6, 0.1),(self.ox1, -0.7, 0.1),(self.ox1, -0.8, 0.1),(self.ox1, -0.9, 0.1),(self.ox1, -1.0, 0.1),
                  
                  (0.3,self.ox, 0.1), (0.4,self.ox, 0.1),( 0.5,self.ox, 0.1),(0.6,self.ox, 0.1),( 0.7,self.ox, 0.1),(0.8,self.ox, 0.1),(0.9,self.ox, 0.1),(1.0,self.ox, 0.1),
                  (-0.3,self.ox, 0.1), (-0.4,self.ox, 0.1),(-0.5,self.ox, 0.1),(-0.6,self.ox, 0.1),(-0.7,self.ox, 0.1),(-0.8,self.ox, 0.1),(-0.9,self.ox, 0.1),(-1.0,self.ox, 0.1),

                (0.3,self.ox1, 0.1), ( 0.4,self.ox1, 0.1),(0.5,self.ox1, 0.1),(0.6, self.ox1, 0.1),(0.7,self.ox1, 0.1),(0.8,self.ox1, 0.1),(0.9,self.ox1, 0.1),( 1.0, self.ox1, 0.1),
                (-0.3,self.ox1, 0.1), ( -0.4,self.ox1, 0.1),(-0.5, self.ox1, 0.1),( -0.6, self.ox1, 0.1),( -0.7,self.ox1, 0.1),( -0.8,self.ox1, 0.1),( -0.9,self.ox1, 0.1),(-1.0,self.ox1, 0.1)]


    def plot(self, ax):
        length=1
        ox, ox1 = self.ox, self.ox1
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






