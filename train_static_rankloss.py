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

def cross_entropy_prob (x,y, weight):
    sum_class = 0
    for c in range(x.shape[1]):
        full_sum = 0
        for n in range(x.shape[0]):
            a = np.exp(x[n, c])
            sum = 0
            for i in range(x.shape[1]):
                b = np.exp(x[n, i])
                sum += b
            full_sum += -np.log(a/sum)*y[n, c]
        sum_class += full_sum * weight[c]
    return sum_class/x.shape[0]

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

def get_traj_feat_time(goal_sink_feat, grid_size, past_traj, future_traj = None):
    feat = np.zeros(goal_sink_feat.shape)
   
    for i in range(goal_sink_feat.shape[0]):
        index = 0
        vals = np.linspace(0, 6, past_traj[i].shape[1])
        for val in vals:
            [x,y] = past_traj[i, :,index]
            index = index+1
            if np.isnan([x,y]).any():
                continue
            feat[i,int(x),int(y)] = val
            
        if future_traj is not None:
            index = 0
            vals = np.linspace(3, 4 ,len(future_traj[i]))
            for val in vals:
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


def normalize_by_max(input):
    offset = input - torch.min(input)
    output = offset / torch.max(offset)
    return output

def normalize_rank(input):
    print("Input is ", torch.max(input.float()- 0.3, torch.zeros((input.shape))))
    # return torch.max(input.float()- 0.3, torch.zeros((input.shape))) 
    return input.float()

def zeroing_loss(c_zero, zeroing_loss): 
    zeroing_loss_r = zeroing_loss.clone()
    for i in range(len(c_zero)):
        zeroing_loss_r[i] = torch.mul(zeroing_loss[i], float(c_zero[i]))
    return(zeroing_loss_r)
""" init param """
#pre_train_weight = 'pre-train-v6-dilated/step1580-loss0.0022763446904718876.pth'
pre_train_weight = None
vis_per_steps = 200
test_per_steps = 100
# resume = "step280-loss0.5675923794730127.pth"
resume = None
exp_name = '7.23'
grid_size = 60
discount = 0.9
lr = 5e-4
n_epoch = 16
batch_size = 8
n_worker = 2
use_gpu = True

loss_criterion = torch.nn.CrossEntropyLoss(weight = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0]).float())
# loss_criterion = torch.nn.CrossEntropyLoss()
# loss_criterion = torch.nn.L1Loss()

if not os.path.exists(os.path.join('exp', exp_name+"robot")):
    os.makedirs(os.path.join('exp', exp_name+"robot"))

host = os.environ['HOSTNAME']
vis2 = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8091)

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
net_robot = RewardNet(n_channels=7, n_classes=1, n_kin = 0)



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
loss_cma = 0
loss_test = 0

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
# opt_robot = torch.optim.SGD(net_robot.parameters(), lr=lr, weight_decay=1e-4)
train_nll_win_robot = vis2.line(X=np.array([[-1, -1]]), Y=np.array([[nll_cma_robot, nll_cma_robot]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train acc robot'))
test_nll_win_robot = vis2.line(X=np.array([-1]), Y=np.array([nll_test_robot]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test acc robot'))
train_loss_win = vis2.line(X=np.array([[-1, -1]]), Y=np.array([[loss_cma, loss_cma]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train loss'))
test_loss_win = vis2.line(X=np.array([-1]), Y=np.array([loss_test]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test loss'))
total_demos = len(train_loader_robot.dataset)

best_test_nll_human = np.inf
best_test_nll_robot = np.inf
prev_past_traj_robot = np.empty([total_demos, grid_size, 2])*np.nan
prev_past_traj_human = np.empty([total_demos, grid_size, 2])*np.nan
prev_predicted_traj_robot = np.empty([total_demos, grid_size, 2])*np.nan
prev_predicted_traj_human = np.empty([total_demos, grid_size, 2])*np.nan


for epoch in range(n_epoch):
    batch_iter = []
    for index, (feat, robot_traj, human_past_traj, robot_past_traj, demo_rank, weights, full_traj) in enumerate(train_loader_robot):
        print("outside loop")
        start = time.time()
        net_robot.train()
        print('main. step {}'.format(step))
        batch_iter.append(feat.shape[0])
        start_full_index = batch_size*index
        end_full_index = batch_size*index+batch_iter[-1]
        # print("Index is!!!! ", start_full_index, end_full_index)
        # print("Shape of feat is", feat.shape)
        ### Initialize the traj feature with just the past trajectory
        # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, past_traj_r)
        # if not np.isnan(prev_predicted_traj_human[start_full_index:end_full_index].all()):
        #     if not np.isnan(prev_past_traj_human[start_full_index:end_full_index]).all():
        #         feat_r[:,5,:] = get_traj_feature(feat_r[:,0], grid_size, prev_past_traj_human[start_full_index:end_full_index], prev_predicted_traj_human[start_full_index:end_full_index])
        feat_test = feat[:,3,:].numpy()
        feat[:,3,:] = get_traj_feature(feat[:,0], grid_size, human_past_traj)
        # goal_svf = feat[:,5,:].float().unsqueeze(dim=1)/6.0
        # feat = feat[:,:5, :]
        # print(feat_test)
        # feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, robot_past_traj)
        nll_list_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, expected_return, zeroing_loss_r = pred(feat, robot_traj, net_robot, n_states, model_robot, grid_size, full_traj)
        # prev_past_traj_robot[start_full_index:end_full_index] = past_traj_r
        # prev_predicted_traj_robot[start_full_index:end_full_index] = auto_pad_future(grid_size, np.array(sampled_trajs_r))
        ### Use perfect information 
        # prev_predicted_traj_robot[start_full_index:end_full_index] = np.array(future_traj_r)
        opt_robot.zero_grad()
        # a hack to enable backprop in pytorch with a vector
        # the normally used loss.backward() only works when loss is a scalar
        c_zero = get_traj_length(robot_traj)/(grid_size*grid_size)
        # c_zero = np.zeros(c_zero.shape)
        for i in range(len(c_zero)):
            denom = r_var_r[i].detach().numpy()/np.linalg.norm(r_var_r[i].detach())
            denom = torch.tensor(denom, dtype = torch.float32)
            zeroing_loss_r[i] = c_zero[i]*zeroing_loss_r[i]/denom
        # traj_rank_weight = normalize_rank(demo_rank)
        traj_rank_weight = weights
        traj_rank_weight = traj_rank_weight.unsqueeze(dim=1)
        traj_rank_weight = traj_rank_weight.unsqueeze(dim=2)
        traj_rank_weight = traj_rank_weight.unsqueeze(dim=3)
        print("Demo rank is ", demo_rank)
    
        # torch.autograd.backward([r_var_r], [-traj_rank_weight.float()*(svf_diff_var_r.float())])  # to maximize, hence add minus sign
        zeroing_loss_criterion = zeroing_loss_r.mean()
        zeroing_loss_full = Variable(zeroing_loss_criterion, requires_grad=True)
        # zeroing_loss_full.backward()
        torch.autograd.backward([r_var_r], [-(svf_diff_var_r.float())])  # to maximize, hence add minus sign
        one_hot_rank = torch.zeros((len(demo_rank)), dtype= torch.long)
        for i in range(len(demo_rank)):
            one_hot_rank[i] = int(demo_rank[i]*10)-2
        # loss = loss_criterion(torch.t(expected_return), one_hot_rank.type(torch.LongTensor))
        # loss_var = Variable(loss, requires_grad=True)
        # loss_var.backward()
        one_hot_new = torch.zeros((len(demo_rank), 9), dtype = torch.float32)
        # for i in range(len(demo_rank)):
        #     x = expected_return[:, i]
        #     for j in x.nonzero():
        #         one_hot_new[i, j] = 1.0/x.nonzero().shape[0]
        for i in range(len(demo_rank)):
            one_hot_new[i, int(demo_rank[i]*10)-2] = 1.0
        # one_hot_new = torch.tensor([0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], dtype = torch.float32)
        # one_hot_new = one_hot_new.repeat(len(demo_rank), 1)
        weight = torch.tensor([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0], dtype = torch.float32)
        loss = cross_entropy_prob(torch.t(expected_return), one_hot_new, weight)
        loss_var = Variable(loss, requires_grad=True)
        # loss_var.backward()
        #### Hack to visualize easy, see TRIBHI
        loss = zeroing_loss_criterion
        # torch.autograd.backward([r_var_r], [-traj_rank_weight.float()*(svf_diff_var_r.float())])
        ### Original loss 
        # torch.autograd.backward([r_var_r], [-svf_diff_var_r.float()])  # to maximize, hence add minus sign
        # print(svf_diff_var_r.shape)
        # half_batch_size = int(np.floor(expected_return.shape[0] / 2))
        # expected_return_var_i = expected_return[:half_batch_size]
        # expected_return_var_j = expected_return[half_batch_size:half_batch_size*2]
        # if half_batch_size == 1:
        #     # print("unsqeeuzed", torch.transpose(expected_return_var_i.unsqueeze(dim=0), 0, 1))
        #     output = torch.cat((expected_return_var_i.unsqueeze(dim=0), (expected_return_var_j.unsqueeze(dim=0))), dim=1)
        # else:
        #     output = torch.cat((expected_return_var_i.unsqueeze(dim=1), expected_return_var_j.unsqueeze(dim=1)), dim=1)
        # loss = zeroing_loss(c_zero, zeroing_loss_r)
        # loss_var = Variable(loss, requires_grad=True)
        # loss_var.backward()

        # rank_cons_i = demo_rank[:half_batch_size]
        # rank_cons_j = demo_rank[half_batch_size:half_batch_size*2]
        # if (half_batch_size == 1):
        #     target = torch.dot(torch.gt(rank_cons_i, rank_cons_j).float(), torch.sub(rank_cons_i, rank_cons_j)).long()
        # else:
        #     target = torch.dot(torch.gt(rank_cons_i, rank_cons_j).float(), torch.sub(rank_cons_i, rank_cons_j).float()).unsqueeze(dim=0) # 0 when i is better, 1 when j is better
        # print(torch.dot(torch.gt(rank_cons_i, rank_cons_j).float(), torch.sub(rank_cons_i, rank_cons_j).float()))
        # print(output, torch.sub(rank_cons_i, rank_cons_j).float())
        # output = expected_return
        # print("Out put and target ", output, demo_rank)
        # loss = loss_criterion(normalize_by_max(output), normalize_rank(demo_rank.float()))
        # loss_var = Variable(loss, requires_grad=True)
        # # loss_var = Variable(demo_rank.type(torch.DoubleTensor), requires_grad = True)
        # print(loss_var/len(demo_rank))
        # # loss_var.backward()
        opt_robot.step()
        # normalized_rank = normalize_rank(demo_rank)
        # print("Normlized ranks is ", normalized_rank)
        # for i in range(len(nll_list_r)):
        #     nll_list_r[i] = nll_list_r[i]*normalized_rank[i]

        nll_r = sum(nll_list_r) / len(nll_list_r)
        print('main. acc {}. took {} s'.format(nll_r, time.time() - start))

        # cma. cumulative moving average. window size < 20
        nll_cma_robot = (nll_r + nll_cma_robot * min(step, 20)) / (min(step, 20) + 1)
        vis2.line(X=np.array([[step, step]]), Y=np.array([[nll_r, nll_cma_robot]]), win=train_nll_win_robot, update='append')
        loss_cma = (loss + loss_cma * min(step, 20)) / (min(step, 20) + 1)
        vis2.line(X=np.array([[step, step]]), Y=np.array([[loss, loss_cma]]), win=train_loss_win, update='append')
        if step % vis_per_steps == 0 and not step ==0 :
            visualize_batch([robot_traj[0]], robot_traj, feat, r_var_r, values_list_r, zeroing_loss_r.float() , step, vis2, grid_size, train=True, policy_sample_list=sampled_trajs_r, rank_list=demo_rank)
            if step == 0:
                step += 1
                continue

        if step % test_per_steps == 0:
        #     # test
            net_robot.eval()
            nll_test_list_human = []
            nll_test_list_robot = []
            for test_index, (feat_r, robot_traj, human_past_traj, robot_past_traj, demo_rank, weights, full_trajs) in enumerate(test_loader_robot):
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
                # feat_r = feat_r[:,:5, :]
                tmp_nll_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, _, _ = pred(feat_r, robot_traj, net_robot, n_states, model_robot, grid_size, full_trajs)
                nll_test_list_robot += tmp_nll_r
            nll_test_robot = sum(nll_test_list_robot) / len(nll_test_list_robot)
            print('main. test nll {}'.format(nll_test_robot))
            vis2.line(X=np.array([step]), Y=np.array([nll_test_robot]), win=test_nll_win_robot, update='append')
            if step % vis_per_steps == 0 and not step ==0:
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
