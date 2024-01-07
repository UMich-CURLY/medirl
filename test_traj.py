import mdp.offroad_grid as offroad_grid
from loader.data_loader_dynamic import OffroadLoader
from torch.utils.data import DataLoader
import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import visdom
import warnings
import logging
import os

warnings.filterwarnings('ignore')
from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated
from network.one_stage_dilated import OneStageDilated
from network.only_env_dilated import OnlyEnvDilated
import torch
from torch.autograd import Variable
import time
from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)


resume = None
exp_name = '6.33robot'
grid_size = 32

resume = 'step240-loss0.024206237815614904.pth'
net = HybridDilated(feat_in_size = 4, feat_out_size = 50)
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://127.0.0.1', port=8098)

def get_traj_feature(goal_sink_feat, grid_size, past_traj, future_traj = None):
    feat = np.zeros(goal_sink_feat.shape)
    for i in range(goal_sink_feat.shape[0]):
        goal_sink_feat_array = np.array(goal_sink_feat.float())
        min_val = np.min(goal_sink_feat_array)
        max_val = np.max(goal_sink_feat_array)
        mean_val = min_val+max_val/2
        index = 0
        for val in np.linspace(6, 5,len(past_traj[i])):
            [x,y] = past_traj[i][index]
            if np.isnan([x,y]).any():
                continue
            feat[i,int(x),int(y)] = val
            index = index+1
        if future_traj is not None:
            index = 0
            for val in np.linspace(3, 4 ,len(future_traj[i])):
                [x,y] = future_traj[i][index]
                if np.isnan([x,y]).any():
                    continue
                feat[i,int(x),int(y)] = val
                index = index+1
    
    return torch.from_numpy(feat)

def traj_interp(c):
    d = c
    iter = len(d) - 1
    added = 0
    i = 0
    while i < iter:
        while np.sqrt((d[i+added,0]-d[i+1+added,0])**2 + (d[i+added,1]-d[i+1+added,1])**2) > np.sqrt(1):
            d = np.insert(d, i+added+1, [0, 0], axis=0)
            if d[i+added+2, 0] - d[i+added, 0] > 0:
                d[i+added+1, 0] = d[i+added, 0] + 1
                d[i+added+1, 1] = d[i+added, 1]
            elif d[i+added+2, 0] - d[i+added, 0] < 0:
                d[i+added+1, 0] = d[i+added, 0] - 1
                d[i+added+1, 1] = d[i+added, 1]
            else:
                d[i+added+1, 0] = d[i+added, 0]
                if d[i+added+2, 1] - d[i+added, 1] > 0:
                    d[i+added+1, 1] = d[i+added, 1] + 1
                elif d[i+added+2, 1] - d[i+added, 1] < 0:
                    d[i+added+1, 1] = d[i+added, 1] - 1
                else:
                    d[i+added+1, 1] = d[i+added, 1]
            added += 1
        i += 1

    connected_map = np.zeros((32, 32))
    for i in range(len(d)):
        connected_map[d[i,1]+1, d[i,0]+1] = 1
    return d

discount = 0.9

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions
n_worker = 8
batch_size = 8
loader = OffroadLoader(grid_size=grid_size, train=False)
loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)

net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp_name, resume))
net.load_state_dict(checkpoint['net_state'])
net.eval()

test_nll_list = []
test_dist_list = []
for step, (feat, past_traj, future_traj, past_traj_h, future_traj_h) in enumerate(loader):
    start = time.time()
    feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, past_traj)
    feat[:,5,:] = get_traj_feature(feat[:,0], grid_size, past_traj_h, future_traj_h)
    nll_list, r_var, svf_diff_var, values_list, sampled_trajs_r, zeroing_loss_r = pred(feat, future_traj, net, n_states, model, grid_size)
    test_nll_list += nll_list
    visualize_batch(past_traj, future_traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=False)
    # print('{}'.format(sum(test_dist_list) / len(test_dist_list)))
nll = sum(test_nll_list) / len(test_nll_list)
# dist = sum(test_dist_list) / len(test_dist_list)
vis.text('nll {}'.format(nll))
# vis.text('distance {}'.format(dist))
