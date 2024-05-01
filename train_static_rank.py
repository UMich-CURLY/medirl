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


def zeroing_loss(c_zero, zeroing_loss): 
    zeroing_loss_r = zeroing_loss.clone()
    for i in range(len(c_zero)):
        zeroing_loss_r[i] = torch.mul(zeroing_loss[i], float(c_zero[i]))
    return(zeroing_loss_r)
""" init param """
#pre_train_weight = 'pre-train-v6-dilated/step1580-loss0.0022763446904718876.pth'
pre_train_weight = None
vis_per_steps = 20
test_per_steps = 20
# resume = "step280-loss0.5675923794730127.pth"
resume = None
exp_name = '6.07'
grid_size = 60
discount = 0.9
lr = 5e-4
n_epoch = 128
batch_size = 8
n_worker = 2
use_gpu = True

loss_criterion = torch.nn.CrossEntropyLoss()


if not os.path.exists(os.path.join('exp', exp_name+"robot")):
    os.makedirs(os.path.join('exp', exp_name+"robot"))

host = os.environ['HOSTNAME']
vis2 = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8098)

# vis = visdom.Visdom(env='main')

model_robot = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model_robot.n_states
n_actions = model_robot.n_actions

print("Train loader")
train_loader_robot = OffroadLoader(grid_size=grid_size, tangent=False)
train_loader_robot = DataLoader(train_loader_robot, num_workers=n_worker, batch_size=batch_size, shuffle=True)
print("test loader")
test_loader_robot = OffroadLoader(grid_size=grid_size, train=False, tangent=False)
test_loader_robot = DataLoader(test_loader_robot, num_workers=n_worker, batch_size=batch_size, shuffle=True)

# net_robot = HybridDilated(feat_in_size = 4, feat_out_size = 50)
# net_robot = OnlyEnvDilated(feat_in_size = 4, feat_out_size = 50)
net_robot = RewardNet(n_channels=4, n_classes=1, n_kin = 0)



# train_loader_human = OffroadLoader(grid_size=grid_size, tangent=False, human = True)
# train_loader_human = DataLoader(train_loader_human, num_workers=n_worker, batch_size=batch_size, shuffle=False)
# test_loader_human = OffroadLoader(grid_size=grid_size, train=False, tangent=False, human = True)
# test_loader_human = DataLoader(test_loader_human, num_workers=n_worker, batch_size=batch_size, shuffle=False)
# net_human = HybridDilated(feat_in_size = 4, feat_out_size = 50)
# net_human = OnlyEnvDilated(feat_in_size = 4, feat_out_size = 50)
#net = OneStageDilated(feat_out_size=25)
step = 0
nll_cma = 0
nll_test = 0

step = 0
nll_cma_human = 0
nll_test_human = 0
nll_cma_robot = 0
nll_test_robot = 0

if resume is None:
    if pre_train_weight is None:
        # net_robot.init_weights()
        # net_human.init_weights()
        pass
    else:
        pre_train_check = torch.load(os.path.join('exp', pre_train_weight))
        net_robot.init_with_pre_train(pre_train_check)
else:
    checkpoint_human = torch.load(os.path.join('exp', exp_name+"human", resume))
    checkpoint_robot = torch.load(os.path.join('exp', exp_name+"robot", resume))
    step = checkpoint_robot['step']
    net_robot.load_state_dict(checkpoint_robot['net_state'])
    nll_cma_human = checkpoint_human['nll_cma']
    nll_cma_robot = checkpoint_robot['nll_cma']
    # opt.load_state_dict(checkpoint['opt_state'])

opt_robot = torch.optim.Adam(net_robot.parameters(), lr=lr, weight_decay=1e-4)

train_nll_win_robot = vis2.line(X=np.array([[-1, -1]]), Y=np.array([[nll_cma_robot, nll_cma_robot]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train acc robot'))
test_nll_win_robot = vis2.line(X=np.array([-1]), Y=np.array([nll_test_robot]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test acc robot'))
""" train """

total_demos = len(train_loader_robot.dataset)

best_test_nll_human = np.inf
best_test_nll_robot = np.inf
prev_past_traj_robot = np.empty([total_demos, grid_size, 2])*np.nan
prev_past_traj_human = np.empty([total_demos, grid_size, 2])*np.nan
prev_predicted_traj_robot = np.empty([total_demos, grid_size, 2])*np.nan
prev_predicted_traj_human = np.empty([total_demos, grid_size, 2])*np.nan


for epoch in range(n_epoch):
    batch_iter = []
    for index, (feat, robot_traj, human_past_traj, robot_past_traj, demo_rank) in enumerate(train_loader_robot):
        print("outside loop")
        start = time.time()
        net_robot.train()
        print('main. step {}'.format(step))
        batch_iter.append(feat.shape[0])
        start_full_index = batch_size*index
        end_full_index = batch_size*index+batch_iter[-1]
        print("Index is!!!! ", start_full_index, end_full_index)
        print("Shape of feat is", feat.shape)
        ### Initialize the traj feature with just the past trajectory
        # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, past_traj_r)
        # if not np.isnan(prev_predicted_traj_human[start_full_index:end_full_index].all()):
        #     if not np.isnan(prev_past_traj_human[start_full_index:end_full_index]).all():
        #         feat_r[:,5,:] = get_traj_feature(feat_r[:,0], grid_size, prev_past_traj_human[start_full_index:end_full_index], prev_predicted_traj_human[start_full_index:end_full_index])
        feat_test = feat[:,3,:].numpy()
        feat[:,3,:] = get_traj_feature(feat[:,0], grid_size, human_past_traj)
        # print(feat_test)
        # feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, robot_past_traj)
        nll_list_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, zeroing_loss_r = pred(feat, robot_traj, net_robot, n_states, model_robot, grid_size)
        # prev_past_traj_robot[start_full_index:end_full_index] = past_traj_r
        # prev_predicted_traj_robot[start_full_index:end_full_index] = auto_pad_future(grid_size, np.array(sampled_trajs_r))
        ### Use perfect information 
        # prev_predicted_traj_robot[start_full_index:end_full_index] = np.array(future_traj_r)
        opt_robot.zero_grad()
        # a hack to enable backprop in pytorch with a vector
        # the normally used loss.backward() only works when loss is a scalar
        c_zero = get_traj_length(robot_traj)/(grid_size*grid_size)
        for i in range(len(c_zero)):
            zeroing_loss_r[i] = c_zero[i]*zeroing_loss_r[i]
        torch.autograd.backward([r_var_r], [-svf_diff_var_r])  # to maximize, hence add minus sign
        # loss = zeroing_loss(c_zero, zeroing_loss_r)
        # loss_var = Variable(loss, requires_grad=True)
        # loss_var.backward()
        opt_robot.step()
        
        nll_r = sum(nll_list_r) / len(nll_list_r)
        print('main. acc {}. took {} s'.format(nll_r, time.time() - start))

        # cma. cumulative moving average. window size < 20
        nll_cma_robot = (nll_r + nll_cma_robot * min(step, 20)) / (min(step, 20) + 1)
        vis2.line(X=np.array([[step, step]]), Y=np.array([[nll_r, nll_cma_robot]]), win=train_nll_win_robot, update='append')

        if step % vis_per_steps == 0 and not step ==0 :
            visualize_batch([robot_traj[0]], robot_traj, feat, r_var_r, values_list_r, svf_diff_var_r , step, vis2, grid_size, train=True, policy_sample_list=sampled_trajs_r)
            if step == 0:
                step += 1
                continue

        if step % test_per_steps == 0:
        #     # test
            net_robot.eval()
            nll_test_list_human = []
            nll_test_list_robot = []
            for test_index, (feat_r, robot_traj, human_past_traj, robot_past_traj) in enumerate(test_loader_robot):
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
                tmp_nll_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, _ = pred(feat_r, robot_traj, net_robot, n_states, model_robot, grid_size)
                nll_test_list_robot += tmp_nll_r
            nll_test_robot = sum(nll_test_list_robot) / len(nll_test_list_robot)
            print('main. test nll {}'.format(nll_test_robot))
            vis2.line(X=np.array([step]), Y=np.array([nll_test_robot]), win=test_nll_win_robot, update='append')
            visualize_batch([robot_traj[0]], robot_traj, feat_r, r_var_r, values_list_r, svf_diff_var_r, step, vis2, grid_size, train=False, policy_sample_list=sampled_trajs_r)
            # print("Robot Traj is ", robot_traj)
            # print("Sampled Traj is ", sampled_trajs_r)
            if nll_test_robot < best_test_nll_robot:
                best_test_nll_robot = nll_test_robot
            state = {'nll_cma': nll_cma_robot, 'test_nll': nll_test_robot, 'step': step, 'net_state': net_robot.state_dict(),
                        'opt_state': opt_robot.state_dict(), 'discount':discount}
            path = os.path.join('exp', exp_name+"robot", 'step{}-loss{}.pth'.format(step, nll_test_robot))
            torch.save(state, path)

        step += 1
