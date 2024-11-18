import torch.nn as nn
import torch
import config
import numpy as np
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from qpth.qp import QPFunction
from model_utils import solver
from util import calculate_all_metrics, calculate_is_not_live_torch

# Indices to make reading the code easier.

EGO_X_IDX = 0
EGO_Y_IDX = 1
EGO_THETA_IDX = 2
EGO_V_IDX = 3
OPP_X_IDX = 4
OPP_Y_IDX = 5
OPP_THETA_IDX = 6
OPP_V_IDX = 7
GOAL_DX_IDX = 8
GOAL_DY_IDX = 9

N_CL = 2
ANGULAR_VEL_IDX = 0
LINEAR_ACCEL_IDX = 1


class BarrierNet(nn.Module):
    # Input features: 8. [ego x, ego y, ego theta, ego v, opp x, opp y, opp theta, opp v]
    # Output controls: 2. [linear vel, angular vel].
    def __init__(self, model_definition, static_obstacles, goal):
        super().__init__()
        self.model_definition = model_definition
        self.mean = torch.from_numpy(np.array(model_definition.input_mean)).to(config.device)
        self.std = torch.from_numpy(np.array(model_definition.input_std)).to(config.device)
        self.static_obstacles = static_obstacles
        self.goal = goal

        # QP Parameters
        self.p1 = 0
        self.p2 = 0

        self.n_features = 8
        
        self.fc1 = nn.Linear(self.n_features, model_definition.nHidden1).double()
        self.fc21 = nn.Linear(model_definition.nHidden1, model_definition.nHidden21).double()
        self.fc22 = nn.Linear(model_definition.nHidden1, model_definition.nHidden22).double()
        if self.model_definition.separate_penalty_for_opp:
            self.fc23 = nn.Linear(model_definition.nHidden1, model_definition.nHidden23).double()
        if self.model_definition.add_liveness_filter:
            self.fc24 = nn.Linear(model_definition.nHidden1, model_definition.nHidden24).double()

        self.fc31 = nn.Linear(model_definition.nHidden21, N_CL).double()
        self.fc32 = nn.Linear(model_definition.nHidden22, N_CL).double()
        if self.model_definition.separate_penalty_for_opp:
            self.fc33 = nn.Linear(model_definition.nHidden23, N_CL).double()
        if self.model_definition.add_liveness_filter:
            self.fc34 = nn.Linear(model_definition.nHidden24, 2).double()

        if model_definition.add_control_limits:
            self.s0 = Parameter(torch.ones(1).cuda()).to(config.device)
            self.s1 = Parameter(torch.ones(1).cuda()).to(config.device)

    

    def forward(self, x, sgn):
        nBatch = x.size(0)

        # Normal FC network.
        x = x.view(nBatch, -1)
        x0 = x*self.std + self.mean
        x = F.relu(self.fc1(x))
        
        x21 = F.relu(self.fc21(x))
        x31 = self.fc31(x21)

        x22 = F.relu(self.fc22(x))
        x32 = self.fc32(x22)
        x32 = 4*nn.Sigmoid()(x32)  # ensure CBF parameters are positive

        if self.model_definition.separate_penalty_for_opp:
            x23 = F.relu(self.fc23(x))
            x33 = self.fc33(x23)
            x33 = 4*nn.Sigmoid()(x33)  # ensure CBF parameters are positive
        else:
            x33 = None

        if self.model_definition.add_liveness_filter:
            x24 = F.relu(self.fc24(x))
            x34 = self.fc34(x24)
            x34 = 4*nn.Sigmoid()(x34)  # ensure CBF parameters are positive
        else:
            x34 = None

        # print(x31, x32, x33, x34)

        # BarrierNet
        x = self.dCBF(x0, x31, x32, x33, x34, sgn, nBatch)
               
        return x

    def dCBF(self, x0, x31, x32, x33, x34, sgn, nBatch):
        px = x0[:,EGO_X_IDX]
        py = x0[:,EGO_Y_IDX]
        if self.model_definition.x_is_d_goal:
            px = self.goal[0] - px
            py = self.goal[1] - py
        theta = x0[:,EGO_THETA_IDX]
        v = x0[:,EGO_V_IDX]
        # dx = x0[:,GOAL_DX_IDX]
        # dy = x0[:,GOAL_DY_IDX]
        Q = Variable(torch.eye(N_CL))
        Q = Q.unsqueeze(0).expand(nBatch, N_CL, N_CL).to(config.device)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        
        obstacles = self.static_obstacles.copy()
        opps = [(x0[:,OPP_X_IDX], x0[:,OPP_Y_IDX], x0[:,OPP_THETA_IDX], x0[:,OPP_V_IDX])]

        G = []
        h = []
        for obs_x, obs_y, r in obstacles:
            R = config.agent_radius + r + config.safety_dist
            dx = (px - obs_x)
            dy = (py - obs_y)

            barrier = dx**2 + dy**2 - R**2
            barrier_dot = 2*dx*v*cos_theta + 2*dy*v*sin_theta
            Lf2b = 2*v**2
            LgLfbu1 = torch.reshape(-2*dx*v*sin_theta + 2*dy*v*cos_theta, (nBatch, 1)) 
            LgLfbu2 = torch.reshape(2*dx*cos_theta + 2*dy*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            obs_h = (torch.reshape(Lf2b + (x32[:,0] + x32[:,1])*barrier_dot + (x32[:,0] * x32[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)

        for opp_x, opp_y, opp_theta, opp_vel in opps:
            R = config.agent_radius + config.agent_radius + config.safety_dist
            if self.model_definition.x_is_d_goal:
                dx, dy = -opp_x, -opp_y
            else:
                dx = (px - opp_x)
                dy = (py - opp_y)
            opp_sin_theta = torch.sin(opp_theta)
            opp_cos_theta = torch.cos(opp_theta)
        
            barrier = dx**2 + dy**2 - R**2
            barrier_dot = 2*dx*(v*cos_theta - opp_vel*opp_cos_theta) + 2*dy*(v*sin_theta - opp_vel*opp_sin_theta)
            Lf2b = 2*(v*v + opp_vel*opp_vel + 2*v*opp_vel*torch.cos(theta - opp_theta))
            LgLfbu1 = torch.reshape(-2*dx*v*sin_theta + 2*dy*v*cos_theta, (nBatch, 1))
            LgLfbu2 = torch.reshape(2*dx*cos_theta + 2*dy*sin_theta, (nBatch, 1))
            obs_G = torch.cat([-LgLfbu1, -LgLfbu2], dim=1)
            obs_G = torch.reshape(obs_G, (nBatch, 1, N_CL))
            penalty = x33 if self.model_definition.separate_penalty_for_opp else x32
            obs_h = (torch.reshape(Lf2b + (penalty[:,0] + penalty[:,1])*barrier_dot + (penalty[:,0] * penalty[:,1])*barrier, (nBatch, 1)))
            G.append(obs_G)
            h.append(obs_h)

        print(len(G), len(h))
        print(G[0].shape, h[0].shape)
        # print(1/0)

        if self.model_definition.add_control_limits:
            G_lims, h_lims = [], []
            for i in range(len(x0)):
                lim_G = Variable(torch.tensor([0.0, 1.0]))
                lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                lim_h = Variable(torch.tensor([config.accel_limit])).to(config.device) + self.s0
                lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                G_lims.append(lim_G)
                h_lims.append(lim_h)

            G_lims = torch.cat(G_lims)
            h_lims = torch.cat(h_lims)
            G.append(G_lims)
            h.append(h_lims)

            G_lims, h_lims = [], []
            for i in range(len(x0)):
                lim_G = Variable(torch.tensor([0.0, -1.0]))
                lim_G = lim_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                lim_h = Variable(torch.tensor([config.accel_limit])).to(config.device) + self.s1
                lim_h = torch.reshape(lim_h, (1, 1)).to(config.device)
                G_lims.append(lim_G)
                h_lims.append(lim_h)

            G_lims = torch.cat(G_lims)
            h_lims = torch.cat(h_lims)
            G.append(G_lims)
            h.append(h_lims)


        # Add in liveness CBF
        if self.model_definition.add_liveness_filter:
            G_live, h_live = [], []
            for i in range(len(x0)):
                # is_not_live will be 1 if it's not live, and 0 if it is live.
                is_not_live = calculate_is_not_live_torch(x0[i,OPP_X_IDX], x0[i,OPP_Y_IDX], theta[i], v[i], x0[i,OPP_THETA_IDX], x0[i,OPP_V_IDX])
                if is_not_live.item() != 0:
                    # print(x0[i,OPP_X_IDX], x0[i,OPP_Y_IDX], theta[i], v[i], x0[i,OPP_THETA_IDX], x0[i,OPP_V_IDX])
                    # print(is_not_live)
                    print("USING LIVENESS FILTER!!!", i)
                    # ego_state = np.array([px[i].cpu().item(), py[i].cpu().item(), theta[i].cpu().item(), v[i].cpu().item()])
                    # opp_state = np.array([(px[i] + x0[i,OPP_X_IDX]).cpu().item(), (py[i] + x0[i,OPP_Y_IDX]).cpu().item(), x0[i,OPP_THETA_IDX].cpu().item(), x0[i,OPP_V_IDX].cpu().item()])
                    # print(ego_state, opp_state)
                    # print(calculate_all_metrics(ego_state, opp_state))
                    # print(1/0)
                else:
                    ego_state = np.array([px[i].cpu().item(), py[i].cpu().item(), theta[i].cpu().item(), v[i].cpu().item()])
                    opp_state = np.array([(px[i] + x0[i,OPP_X_IDX]).cpu().item(), (py[i] + x0[i,OPP_Y_IDX]).cpu().item(), x0[i,OPP_THETA_IDX].cpu().item(), x0[i,OPP_V_IDX].cpu().item()])
                    metrics = calculate_all_metrics(ego_state, opp_state)
                    if not metrics[-1]:
                        print(x0[i,OPP_X_IDX], x0[i,OPP_Y_IDX], theta[i], v[i], x0[i,OPP_THETA_IDX], x0[i,OPP_V_IDX])
                        print(is_not_live)
                        print("UNLIVE!", metrics)
                        print(ego_state)
                        print(opp_state)
                        print(1/0)


                # If we're going faster, use the speeding up CBF.
                # Otherwise, use the slowing down CBF.
#                if v[i] > x0[i][OPP_V_IDX]:
                if False: # For now force it to slow down, just for testing purposes.
                    # ego_v - zeta * opp_v >= 0.0
                    # b(x) = opp_v - zeta * ego_v
                    barrier = v[i] - config.zeta * x0[i][OPP_V_IDX]
                else:
                    # opp_v - zeta * ego_v >= 0.0
                    # b(x) = opp_v - zeta * ego_v
                    barrier = x0[i][OPP_V_IDX] - config.zeta * v[i]

                # u(x) <= p(x) * b(x)
                is_not_live -= 0.00001
                # is_not_live = torch.tensor(0.99999)
                live_G = Variable(is_not_live*torch.tensor([0.0, 1.0]).to(config.device)).to(config.device)
                live_G = live_G.unsqueeze(0).expand(1, 1, N_CL).to(config.device)
                live_h = torch.reshape(is_not_live*(x34[i,0])*(barrier + x34[i,1]), (1, 1)).to(config.device)
                if is_not_live.item() > 0:
                    print(live_G, live_h)
                    print("\t", barrier, x34[i,0], x34[i,1])

                G_live.append(live_G)
                h_live.append(live_h)

            G_live = torch.cat(G_live)
            h_live = torch.cat(h_live)
            print(G_live.shape, h_live.shape)
            G.append(G_live)
            h.append(h_live)


        
        print(len(G), len(h))
        
        G = torch.cat(G, dim=1).to(config.device)
        h = torch.cat(h, dim=1).to(config.device)
        num_ineq = len(obstacles) + len(opps) + self.model_definition.add_control_limits * 2 + self.model_definition.add_liveness_filter
        assert(G.shape == (nBatch, num_ineq, N_CL))
        assert(h.shape == (nBatch, num_ineq))
        e = Variable(torch.Tensor()).to(config.device)
        
        # label_std, label_mean = np.array(self.model_definition.label_std), np.array(self.model_definition.label_mean)
        # print("Reference controls:", x31.cpu() * label_std + label_mean)

        if self.training or sgn == 1:    
            x = QPFunction(verbose = 0)(Q.double(), x31.double(), G.double(), h.double(), e, e)
        else:
            self.p1 = x32[0,0]
            self.p2 = x32[0,1]
            print(x31[0].cpu())
            try:
                x = solver(Q[0].double(), x31[0].double(), G[0].double(), h[0].double())
            except Exception as e:
                print("ERROR WHEN SOLVING FOR OPTIMIZER, USING REFERENCE CONTROL INSTEAD:", x31)
                x = x31[0].cpu()
       
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
