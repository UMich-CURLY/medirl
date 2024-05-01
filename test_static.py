import mdp.offroad_grid as offroad_grid
from loader.data_loader_static import OffroadLoader
from torch.utils.data import DataLoader
import numpy as np
import visdom

from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated

from torch.autograd import Variable
import torch
import time
from multiprocessing import Pool
import os
from maxent_irl_social import visualize_batch
from network.reward_net import RewardNet
from IPython import embed
# initialize param
grid_size = 60

discount = 0.9
batch_size = 3
n_worker = 2
#exp = '6.24'
#resume = 'step700-loss0.6980162681374217.pth'
#net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

exp_name = '6.07robot'
resume  = 'step1880-loss3.8473885302669486.pth'
net = RewardNet(n_channels=4, n_classes=1, n_kin = 0)
# self.net.init_weights()
checkpoint = torch.load(os.path.join('exp', exp_name, resume))
net.load_state_dict(checkpoint['net_state'])

# def rl(future_traj_sample, r_sample, model, grid_size):
#     svf_demo_sample = model.find_demo_svf(future_traj_sample)
#     values_sample = model.find_optimal_value(r_sample, 0.01)
#     policy = model.find_stochastic_policy(values_sample, r_sample)
#     svf_sample = model.find_svf(future_traj_sample, policy)
#     svf_diff_sample = svf_demo_sample - svf_sample
#     # (1, n_feature, grid_size, grid_size)
#     svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
#     svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
#     nll_sample = model.compute_nll(policy, future_traj_sample)
#     dist_sample = model.compute_hausdorff_loss(policy, future_traj_sample, n_samples=1000)
#     return nll_sample, svf_diff_var_sample, values_sample, dist_sample

def get_traj_feature(goal_sink_feat, grid_size, past_traj, future_traj = None):
    feat = np.zeros(goal_sink_feat.shape)
    past_lengths, past_traj = get_traj_length_unique(past_traj)
    if future_traj is not None:
        future_lengths, future_traj = get_traj_length_unique(future_traj)
    for i in range(goal_sink_feat.shape[0]):
        goal_sink_feat_array = np.array(goal_sink_feat.float())
        min_val = np.min(goal_sink_feat_array)
        max_val = np.max(goal_sink_feat_array)
        mean_val = min_val+max_val/2
        index = 0
        for val in np.linspace(6, 5, past_lengths[i]):
            [x,y] = past_traj[i][index]
            if np.isnan([x,y]).any():
                continue
            feat[i,int(x),int(y)] = 6
            index = index+1
        if future_traj is not None:
            index = 0
            for val in np.linspace(3, 4 ,future_lengths[i]):
                [x,y] = future_traj[i][index]
                if np.isnan([x,y]).any():
                    continue
                feat[i,int(x),int(y)] = val
                index = index+1
    
    return torch.from_numpy(feat)

def get_traj_length(traj):
    lengths = []
    for i in range(len(traj)):
        traj_sample = traj[i].numpy()  # choose one sample from the batch
        traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        lengths.append(len(traj_sample))
    return np.array(lengths)

def get_traj_length_unique(traj):
    lengths = []
    traj_list_full = []
    for i in range(len(traj)):
        traj_sample = traj[i].numpy()  # choose one sample from the batch
        traj_sample = traj_sample.T
        traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        traj_list = []
        for j in range(len(traj_sample)):
            if list(traj_sample[j]) not in traj_list:
                traj_list.append([traj_sample[j][0], traj_sample[j][1]])
        lengths.append(len(traj_list))
        traj_list_full.append(traj_list)
    return np.array(lengths), traj_list_full


# def pred(feat, future_traj, net, n_states, model, grid_size):
#     n_sample = feat.shape[0]
#     feat = feat.float()
#     feat_var = Variable(feat)
#     r_var = net(feat_var)

#     result = []
#     pool = Pool(processes=n_sample)
#     for i in range(n_sample):
#         r_sample = r_var[i].data.numpy().squeeze().reshape(n_states)
#         future_traj_sample = future_traj[i].numpy()  # choose one sample from the batch
#         future_traj_sample = future_traj_sample[~np.isnan(future_traj_sample).any(axis=1)]  # remove appended NAN rows
#         future_traj_sample = future_traj_sample.astype(np.int64)
#         result.append(pool.apply_async(rl, args=(future_traj_sample, r_sample, model, grid_size)))
#     pool.close()
#     pool.join()
#     # extract result and stack svf_diff
#     nll_list = [result[i].get()[0] for i in range(n_sample)]
#     dist_list = [result[i].get()[3] for i in range(n_sample)]
#     svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
#     values_list = [result[i].get()[2] for i in range(n_sample)]
#     svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
#     return nll_list, r_var, svf_diff_var, values_list, dist_list

def rl(traj_sample, r_sample, model, grid_size):
    svf_demo_sample = model.find_demo_svf(traj_sample)
    values_sample = model.find_optimal_value(r_sample, 0.1)
    policy = model.find_stochastic_policy(values_sample, r_sample)
    ### Can change to sampling longer trajectories
    sampled_traj = model.traj_sample(policy, traj_sample.shape[0], traj_sample[0,0], traj_sample[0,1])
    svf_sample = model.find_svf(traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    zeroing_loss = np.where(svf_sample>0,svf_demo_sample + svf_sample, 0.0)
    # (1, n_feature, grid_size, grid_size)
    svf_diff_sample = svf_diff_sample.reshape(1, 1, grid_size, grid_size)
    zeroing_loss_sample = zeroing_loss.reshape(1, 1, grid_size, grid_size)
    svf_diff_var_sample = Variable(torch.from_numpy(svf_diff_sample).float(), requires_grad=False)
    zeroing_loss_var_sample = Variable(torch.from_numpy(zeroing_loss_sample).float(), requires_grad=False)
    # embed()
    nll_sample = model.compute_nll(policy, traj_sample)
    return nll_sample, svf_diff_var_sample, values_sample, sampled_traj, zeroing_loss_var_sample


def pred(feat, traj, net, n_states, model, grid_size):
    n_sample = feat.shape[0]
    feat = feat.float()
    feat_var = Variable(feat)
    r_var = net(feat_var)
    result = []
    pool = Pool(processes=n_sample)
    for i in range(n_sample):
        r_sample = r_var[i].data.numpy().squeeze().reshape(n_states)
        traj_sample = traj[i].numpy()  # choose one sample from the batch
        traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        traj_sample = traj_sample.astype(np.int64)
        # result.append(rl(traj_sample, r_sample, model, grid_size))
        result.append(pool.apply_async(rl, args=(traj_sample, r_sample, model, grid_size)))
    pool.close()
    pool.join()
    # extract result and stack svf_diff
    # embed()
    nll_list = [result[i].get()[0] for i in range(n_sample)]
    svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
    values_list = [result[i].get()[2] for i in range(n_sample)]
    policy_sample_list = [result[i].get()[3] for i in range(n_sample)]
    zeroing_loss_list = [result[i].get()[4] for i in range(n_sample)]
    svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
    zeroing_loss = torch.cat(zeroing_loss_list, dim=0)
    return nll_list, r_var, svf_diff_var, values_list, policy_sample_list, zeroing_loss


host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8098)
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = OffroadLoader(grid_size=grid_size, train=False)
loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)


net.eval()

nll_test_list_robot = []
test_dist_list = []
step = 1
for step, (feat_r, robot_traj, human_past_traj, robot_past_traj) in enumerate(loader):
    # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, past_traj_r)
    # if not np.isnan(prev_predicted_traj_human[start_full_index:end_full_index].all()):
    #     if not np.isnan(prev_past_traj_human[start_full_index:end_full_index]).all():
    #         feat_r[:,5,:] = get_traj_feature(feat_r[:,0], grid_size, prev_past_traj_human[start_full_index:end_full_index], prev_predicted_traj_human[start_full_index:end_full_index])
    feat_r[:,3,:] = get_traj_feature(feat_r[:,0], grid_size, human_past_traj)
    # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, robot_past_traj)
    # feat_h[:,4,:] = get_traj_feature(feat_h[:,0], grid_size, past_traj_h)
    # if not np.isnan(prev_predicted_traj_robot[start_full_index:end_full_index]).all():
    #     if not np.isnan(prev_past_traj_robot[start_full_index:end_full_index]).all():
    #         feat_h[:,5,:] = get_traj_feature(feat_h[:,0], grid_size, prev_past_traj_robot[start_full_index:end_full_index], prev_predicted_traj_robot[start_full_index:end_full_index])
    # feat_h[:,4,:] = get_traj_feature(feat_h[:,0], grid_size, past_traj_r, future_traj_r)
    tmp_nll_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, _ = pred(feat_r, robot_traj, net, n_states, model, grid_size)
    print("Learned traj is ", sampled_trajs_r)
    nll_test_list_robot += tmp_nll_r
    visualize_batch([robot_traj[0]], robot_traj, feat_r, r_var_r, values_list_r, svf_diff_var_r, step, vis, grid_size, train=False, policy_sample_list=sampled_trajs_r)

nll_test_robot = sum(nll_test_list_robot) / len(nll_test_list_robot)
print('main. test nll {}'.format(nll_test_robot))
visualize_batch([robot_traj[0]], robot_traj, feat_r, r_var_r, values_list_r, svf_diff_var_r, step, vis, grid_size, train=False, policy_sample_list=sampled_trajs_r)

