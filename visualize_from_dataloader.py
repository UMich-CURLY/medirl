import mdp.offroad_grid as offroad_grid
from loader.data_loader_rank import OffroadLoader
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
from network.reward_net import RewardNet
import torch
from torch.autograd import Variable
import time
from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)
def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader):
        if index == target:
            return data
    return None
pre_train_weight = None
vis_per_steps = 1
test_per_steps = 10
# resume = "step280-loss0.5675923794730127.pth"
resume = None
exp_name = '6.43'
grid_size = 60
discount = 0.9
lr = 5e-3
n_epoch = 32
batch_size = 1
n_worker = 2
use_gpu = True
print("Train loader")
train_loader_robot = OffroadLoader(grid_size=grid_size, tangent=False, train = False)
train_loader_robot = DataLoader(train_loader_robot, num_workers=n_worker, batch_size=batch_size, shuffle=False)

embed()
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8093)
counter = 0
prev_demo = "demo_0"
for index, (feat, robot_traj, human_past_traj, robot_past_traj, demo_rank, weight, full_trajs) in enumerate(train_loader_robot):
    # for i in range(full_trajs.shape[1]):
    #     traj_sample = full_trajs[0,i].numpy()  # choose one sample from the batch
    #     # traj_sample = robot_traj[0].numpy()
    #     traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
    #     traj_sample = traj_sample.astype(np.int64)
    #     if len(traj_sample) == 0:
    #         continue
    #     image_fol = train_loader_robot.dataset.data_list[counter]
    #     print("Out here image fol is ", image_fol)
    #     item = int(image_fol.split('/')[-1])
    #     demo = image_fol.split('/')[-2]

    #     overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
    #     overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
    #     if counter % vis_per_steps == 0:
    #         overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
    #         overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
    #         vis.heatmap(X=overlay_map,
    #                 opts=dict(colormap='Electric', title='{}, step {} ix {}'.format(demo, item, i)))
    image_fol = train_loader_robot.dataset.data_list[counter]
    print("Out here image fol is ", image_fol)
    item = int(image_fol.split('/')[-1])
    demo = image_fol.split('/')[-2]
    if prev_demo != demo:
        # embed()
        prev_demo = demo    
    traj_sample = robot_traj[0].numpy()
    traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
    traj_sample = traj_sample.astype(np.int64)
    human_pos = human_past_traj[0,:,-1].numpy()
    min_dist = 1000
    for point in traj_sample:
        dist = np.linalg.norm(human_pos - point, axis=0)
        min_dist = min(min_dist, dist)
    weight = 6.0-min_dist*0.1
    min_dist = min_dist*0.1
    if min_dist <1.0:
        weight = 1.0
    else:
        weight = 0.7
    with open(image_fol + '/weight.txt', 'w') as f:
        f.write(str(weight))
        f.close()
    if len(traj_sample) == 0:
        continue
    overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
    overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
    if counter % vis_per_steps == 0:
        overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
        overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
        vis.heatmap(X=overlay_map,
                opts=dict(colormap='Electric', title='{}, step {} traj'.format(demo, item)))
        print("Traj sampple is ", traj_sample)
    counter += 1
