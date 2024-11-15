import torch
import config
import numpy as np
from models import FCNet, BarrierNet, ModelDefinition
from util import calculate_liveliness


class BlankController:
    def __init__(self):
        pass

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        pass
    
    def make_step(self, timestamp, initial_state):
        return np.array([[0.0], [0.0]])
