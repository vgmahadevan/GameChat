import os
import torch.nn as nn
import torch
import config
import json
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from qpth.qp import QPFunction
from cvxopt import solvers, matrix
from dataclasses import dataclass, asdict
from typing import Optional

# Indices to make reading the code easier.

N_FEATURES = 8
EGO_X_IDX = 0
EGO_Y_IDX = 1
EGO_THETA_IDX = 2
EGO_V_IDX = 3
OPP_X_IDX = 4
OPP_Y_IDX = 5
OPP_THETA_IDX = 6
OPP_V_IDX = 7

N_CL = 2
LINEAR_ACCEL_IDX = 0
ANGULAR_VEL_IDX = 1


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
    nHidden1: int
    nHidden21: int
    nHidden22: Optional[int]
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
            return ModelDefinition(**data)


class BarrierNet(nn.Module):
    # Input features: 8. [ego x, ego y, ego theta, ego v, opp x, opp y, opp theta, opp v]
    # Output controls: 2. [linear vel, angular vel].
    def __init__(self, model_definition, static_obstacles):
        super().__init__()
        self.mean = torch.from_numpy(np.array(model_definition.input_mean)).to(config.device)
        self.std = torch.from_numpy(np.array(model_definition.input_std)).to(config.device)
        self.static_obstacles = static_obstacles

        # QP Parameters
        self.p1 = 0
        self.p2 = 0
        
        self.fc1 = nn.Linear(N_FEATURES, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc22 = nn.Linear(model_definition.nHidden1, model_definition.nHidden22).double()
        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()
        self.fc32 = nn.Linear(model_definition.nHidden22, N_CL).double()
    

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        
        x21 = F.relu(self.fc21(x))
        x22 = F.relu(self.fc22(x))
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, sgn, nBatch):
        Q = Variable(torch.eye(N_CL))
        Q = Q.unsqueeze(0).expand(nBatch, N_CL, N_CL).to(config.device)
        px = x0[:,EGO_X_IDX]
        py = x0[:,EGO_Y_IDX]
        theta = x0[:,EGO_THETA_IDX]
        v = x0[:,EGO_V_IDX]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        # obstacles = self.static_obstacles[:5].copy()
        obstacles = self.static_obstacles.copy()
        obstacles.append((x0[:,OPP_X_IDX], x0[:,OPP_Y_IDX], config.agent_radius))

        G = []
        h = []
        for obs_x, obs_y, r in obstacles:
            R = config.agent_radius + r + config.safety_dist
            barrier = (px - obs_x)**2 + (py - obs_y)**2 - R**2
            barrier_dot = 2*(px - obs_x)*v*cos_theta + 2*(py - obs_y)*v*sin_theta
            Lf2b = 2*v**2
            LgLfbu1 = torch.reshape(-2*(px - obs_x)*v*sin_theta + 2*(py - obs_y)*v*cos_theta, (nBatch, 1)) 
            LgLfbu2 = torch.reshape(2*(px - obs_x)*cos_theta + 2*(py - obs_y)*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            obs_h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0]*x32[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)
        
        G = torch.cat(G, dim=1).to(config.device)
        h = torch.cat(h, dim=1).to(config.device)
        assert(G.shape == (nBatch, len(obstacles), N_CL))
        assert(h.shape == (nBatch, len(obstacles)))
        e = Variable(torch.Tensor()).to(config.device)
            
        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x


class FCNet(nn.Module):
    def __init__(self, model_definition):
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()


    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x = F.relu(self.fc1(x))
        x21 = F.relu(self.fc21(x))
        x31 = self.fc31(x21)
        
        return x31
