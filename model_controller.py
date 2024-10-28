import torch
import config
import numpy as np
from models import FCNet, BarrierNet, ModelDefinition, N_FEATURES
from util import calculate_liveliness


class ModelController:
    def __init__(self, model_definition_filepath, static_obs):
        self.model_definition = ModelDefinition.from_json(model_definition_filepath)
        self.u_ori = []
        self.liveliness = []
        if self.model_definition.is_barriernet:
            self.model = BarrierNet(self.model_definition, static_obs).to(config.device)
        else:
            self.model = FCNet(self.model_definition).to(config.device)
        self.model.load_state_dict(torch.load(self.model_definition.weights_path))
        self.model.eval()

    def initialize_controller(self, env):
        pass

    def reset_state(self, initial_state, opp_state):
        self.initial_state = initial_state
        self.opp_state = opp_state
    
    def make_step(self, initial_state):
        self.initial_state = initial_state
        model_input_original = np.append(self.initial_state, self.opp_state)
        model_input = (model_input_original - self.model_definition.input_mean) / self.model_definition.input_std

        self.liveliness.append(calculate_liveliness(self.initial_state, self.opp_state))

        with torch.no_grad():
            model_input = torch.autograd.Variable(torch.from_numpy(model_input), requires_grad=False)
            model_input = torch.reshape(model_input, (1, N_FEATURES)).to(config.device)
            model_output = self.model(model_input, 0)
            if self.model_definition.is_barriernet:
                model_output = np.array([model_output[0], model_output[1]])
            else:
                model_output = model_output.reshape(-1).cpu().detach().numpy()

        output = model_output * self.model_definition.label_std + self.model_definition.label_mean
        output = output.reshape(-1, 1)
        self.u_ori.append(output)
        return output
