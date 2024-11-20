import os
import config
import json
import numpy as np
from qpth.qp import QPFunction
from cvxopt import solvers, matrix
from dataclasses import dataclass, asdict
from typing import Optional


def solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())
    
    solvers.options['show_progress'] = False
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    
    return sol['x']


@dataclass
class ModelDefinition:
    is_barriernet: bool
    weights_path: Optional[str]
    nInputs: int
    nHidden1: int
    nHidden21: int
    nHidden22: Optional[int]
    nHidden23: Optional[int]
    nHidden24: Optional[int]
    input_mean: float
    input_std: float
    label_mean: float
    label_std: float
    add_control_limits: bool # Default = false
    add_liveness_filter: bool # Default = false
    separate_penalty_for_opp: bool # Default = false
    x_is_d_goal: bool # Default = false
    vx_vy_inputs: bool
    ax_ay_output: bool
    add_liveness_as_input: bool


    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f)
    
    @staticmethod
    def from_json(path: str):
        with open(path, 'r') as f:
            data = json.load(f)
            path_dir = os.path.dirname(path)
            weights_path = os.path.join(path_dir, data['weights_path'])
            data['weights_path'] = weights_path
            if 'nInputs' not in data:
                data['nInputs'] = 8
            if 'add_control_limits' not in data:
                data['add_control_limits'] = False
            if 'add_liveness_filter' not in data:
                data['add_liveness_filter'] = False
            if 'separate_penalty_for_opp' not in data:
                data['separate_penalty_for_opp'] = False
            if 'x_is_d_goal' not in data:
                data['x_is_d_goal'] = False
            if 'vx_vy_inputs' not in data:
                data['vx_vy_inputs'] = False
            if 'add_liveness_as_input' not in data:
                data['add_liveness_as_input'] = False
            if 'ax_ay_output' not in data:
                data['ax_ay_output'] = False
            return ModelDefinition(**data)
