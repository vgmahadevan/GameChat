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
    include_goal: bool
    nHidden1: int
    nHidden21: int
    nHidden22: Optional[int]
    nHidden23: Optional[int]
    input_mean: float
    input_std: float
    label_mean: float
    label_std: float

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
            if 'include_goal' not in data:
                data['include_goal'] = False
            return ModelDefinition(**data)