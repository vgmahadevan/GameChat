import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from qpth.qp import QPFunction
from cvxopt import solvers, matrix
import config

def solver(Q, p, G, h):
    mat_Q = matrix(Q.cpu().numpy())
    mat_p = matrix(p.cpu().numpy())
    mat_G = matrix(G.cpu().numpy())
    mat_h = matrix(h.cpu().numpy())
    
    solvers.options['show_progress'] = False
    
    sol = solvers.qp(mat_Q, mat_p, mat_G, mat_h)
    
    return sol['x']


class BarrierNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.bn = bn
        self.nCls = nCls
        self.mean = torch.from_numpy(mean).to(device)
        self.std = torch.from_numpy(std).to(device)
        self.device = device
        self.p1 = 0
        self.p2 = 0
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)
            self.bn22 = nn.BatchNorm1d(nHidden22)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc22 = nn.Linear(nHidden1, nHidden22).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()
        self.fc32 = nn.Linear(nHidden22, nCls).double()

        # QP params.
        # from previous layers
    

    def add_obstacles(self, obstacles):
        self.static_obstacles = obstacles


    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        x22 = F.relu(self.fc22(x))
        if self.bn:
            x22 = self.bn22(x22)
        
        x31 = self.fc31(x21)
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive
        
        # BarrierNet
        x = self.dCBF(x0, x31, x32, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, sgn, nBatch):

        Q = Variable(torch.eye(self.nCls))
        Q = Q.unsqueeze(0).expand(nBatch, self.nCls, self.nCls).to(self.device)
        xOriginal = x0 * self.std + self.mean
        px = xOriginal[:,0]
        py = xOriginal[:,1]
        theta = xOriginal[:,2]
        oppx = xOriginal[:,3]
        oppy = xOriginal[:,4]
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        LgLfbu1_obs = []
        LgLfbu2_obs = []
        barriers_obs = []
        # Consider all static obstacles, and also other agents as static obstacles.
        obstacles = self.static_obstacles.copy()
        obstacles.append((oppx, oppy, config.agent_radius))
        for obs in obstacles:
            obs_dx, obs_dy = px - obs[0], py - obs[1]
            obs_r = obs[2]
            # barrier = obs_dx**2 + obs_dy**2 - config.r**2
            # barrier_dot = 2*obs_dx*v*cos_theta + 2*obs_dy*v*sin_theta

            # f(x) = [v * cos(theta), v * sin(theta), 0, 0]
            # g(x) = [0, 0, 1, 1]

            # grad_Lfh(x) = [
            #     2*v*cos(theta),
            #     2*v*sin(theta),
            #     -2*obs_dx*v*sin_theta + 2*obs_dy*v*cos_theta
            #     2*obs_dx*cos_theta + 2*obs_dy*sin_theta,
            # ]

            # Lf2b = 2*v**2
            # LgLfbu1 = torch.reshape(-2*(px - obs[0])*v*sin_theta + 2*(py - obs[1])*v*cos_theta, (nBatch, 1)) 
            # LgLfbu2 = torch.reshape(2*(px - obs[0])*cos_theta + 2*(py - obs[1])*sin_theta, (nBatch, 1))

            # Barrier function.
            barrier = obs_dx**2 + obs_dy**2 - obs_r**2

            # Merging example.
            # hdot <= p(z)(h)
            # hdot = vkp - vk - psi*u
            # vkp - vk - psi*u <= p(z)(h)
            # - psi*u <= vk - vkp + p(z)(h)

            # Deadlock example.
            # hdot <= (p1(z) + p2(z)) * b(x)
            # hdot = 2*obs_dx*v*cos_theta + 2*obs_dy*v*sin_theta
            # 2*obs_dx*v*cos_theta + 2*obs_dy*v*sin_theta <= (p1(z) + p2(z)) * b(x)
            # v * (2*obs_dx*cos(theta) + 2*obs_dy*sin(theta)) <= (p1(z) + p2(z)) * b(x)

            LgLfbu1 = (2*obs_dx*cos_theta) + (2*obs_dy*sin_theta)
            LgLfbu2 = 0.0

            LgLfbu1_obs.append([LgLfbu1, LgLfbu2])
            LgLfbu2_obs.append(LgLfbu2)
            barriers_obs.append(barrier)

        # Add CBF for the other agent.


        num_obs = len(barriers_obs)
        G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
        G = torch.reshape(G, (nBatch, num_obs, self.nCls)).to(self.device)     
        h = (torch.reshape((x32[:,0]*x32[:,1])*barriers_obs, (nBatch, num_obs))).to(self.device) 
        e = Variable(torch.Tensor()).to(self.device)
            
        if self.training or sgn == 1:    
            # pass
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
        
        return x




class FCNet(nn.Module):
    def __init__(self, nFeatures, nHidden1, nHidden21, nHidden22, nCls, mean, std, device, bn):
        super().__init__()
        self.nFeatures = nFeatures
        self.nHidden1 = nHidden1
        self.nHidden21 = nHidden21
        self.nHidden22 = nHidden22
        self.nCls = nCls
        self.mean = mean
        self.std = std
        self.device = device
        self.bn = bn
        
        
        # Normal BN/FC layers.
        if bn:
            self.bn1 = nn.BatchNorm1d(nHidden1)
            self.bn21 = nn.BatchNorm1d(nHidden21)

        self.fc1 = nn.Linear(nFeatures, nHidden1).double()
        self.fc21 = nn.Linear(nHidden1, nHidden21).double()
        self.fc31 = nn.Linear(nHidden21, nCls).double()


    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        
        x = F.relu(self.fc1(x))
        if self.bn:
            x = self.bn1(x)
        
        x21 = F.relu(self.fc21(x))
        if self.bn:
            x21 = self.bn21(x21)
        
        x31 = self.fc31(x21)
        
        return x31
        
        
        