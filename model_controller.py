import torch
import config
import numpy as np
from model_utils import ModelDefinition
from models import FCNet, BarrierNet, BarrierNetDOpp
from util import calculate_liveliness


class ModelController:
    def __init__(self, model_definition_filepath, goal, static_obs):
        self.model_definition = ModelDefinition.from_json(model_definition_filepath)
        self.goal = goal
        if self.model_definition.is_barriernet:
            if config.bn_model == "barriernet":
                self.model = BarrierNet(self.model_definition, static_obs).to(config.device)
            else:
                self.model = BarrierNetDOpp(self.model_definition, static_obs).to(config.device)
        else:
            self.model = FCNet(self.model_definition).to(config.device)
        print(self.model_definition.weights_path)
        self.model.load_state_dict(torch.load(self.model_definition.weights_path))
        self.model.eval()

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        self.initial_state = initial_state
        self.opp_state = opp_state
    
    def make_step(self, timestamp, initial_state):
        self.initial_state = initial_state
        model_input_original = np.append(self.initial_state, self.opp_state)
        if self.model_definition.include_goal:
            dx = self.goal[0] - self.initial_state[0]
            dy = self.goal[1] - self.initial_state[1]
            model_input_original = np.append(model_input_original, [dx, dy])
        model_input = (model_input_original - self.model_definition.input_mean) / self.model_definition.input_std

        with torch.no_grad():
            model_input = torch.autograd.Variable(torch.from_numpy(model_input), requires_grad=False)
            model_input = torch.reshape(model_input, (1, self.model.n_features)).to(config.device)
            model_output = self.model(model_input, 0)
            if self.model_definition.is_barriernet:
                model_output = np.array([model_output[0], model_output[1]])
            else:
                model_output = model_output.reshape(-1).cpu().detach().numpy()

        output = model_output * self.model_definition.label_std + self.model_definition.label_mean
        print("Outputted controls:", output)
        output = output.reshape(-1, 1)
        return output
