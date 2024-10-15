import mdp.offroad_grid as offroad_grid
from loader.data_loader_rank import OffroadLoader
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
from PIL import Image, ImageFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.animation as animation
# initialize param
grid_size = 60
# ImageFile.LOAD_TRUNCATED_IMAGES = True
discount = 0.9
batch_size = 1
n_worker = 2
#exp = '6.24'
#resume = 'step700-loss0.6980162681374217.pth'
#net = HybridDilated(feat_out_size=25, regression_hidden_size=64)

exp_name = '7.32robot'
resume  = 'step22100-loss0.9113-train_loss0.7555.pth'
net = RewardNet(n_channels=7, n_classes=1, n_kin = 0)
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

def make_plot_and_save(data, filename):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(data, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Optional: Add a colorbar to the side

    # Save the heatmap to a temporary file
    plt.savefig(filename, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
    return

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


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
            index = index+1
            if np.isnan([x,y]).any():
                continue
            feat[i,int(x),int(y)] = 6
            
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
        vals = np.linspace(0,6, past_traj[i].shape[1])

        # print(past_traj[i].shape)
        # print(vals)
        for val in vals:
            [x,y] = past_traj[i, :,index].float()
            # print("XY", x,y, val)
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
    # sampled_traj = model.traj_sample(policy, 20, traj_sample[0,0], traj_sample[0,1])
    svf_sample = model.find_svf(traj_sample, policy)
    svf_diff_sample = svf_demo_sample - svf_sample
    zeroing_loss = np.where(np.round(svf_sample+svf_demo_sample,3)>0.0, 1.0, 0.0)
    zeroing_loss = zeroing_loss.astype(np.float32)
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
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8092)
model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

loader = OffroadLoader(grid_size=grid_size, train=False)
loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)
loss_cma = 0
train_loss_win = vis.line(X=np.array([-1]), Y=np.array([loss_cma]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train loss'))
def compute_return(reward, traj):
        total_reward = 0
        discount = 1
        for xy in traj:
            total_reward += discount * reward[:, xy[0], xy[1]]
            discount *= 0.9
        #total_reward = total_reward / traj.shape[0] * 100

        return total_reward

net.eval()

nll_test_list_robot = []
test_dist_list = []
step = 1
returns_list = []
im_array = []
plt_frames = []
prev_demo = "demo_0"

for step, (feat_r, robot_traj, human_past_traj, robot_past_traj, demo_rank, weights,  full_trajs) in enumerate(loader):
    # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, past_traj_r)
    # if not np.isnan(prev_predicted_traj_human[start_full_index:end_full_index].all()):
    #     if not np.isnan(prev_past_traj_human[start_full_index:end_full_index]).all():
    #         feat_r[:,5,:] = get_traj_feature(feat_r[:,0], grid_size, prev_past_traj_human[start_full_index:end_full_index], prev_predicted_traj_human[start_full_index:end_full_index])
    image_fol = loader.dataset.data_list[step]
    print("Out here image fol is ", image_fol)
    item = int(image_fol.split('/')[-1])
    demo = image_fol.split('/')[-2]
    if prev_demo != demo:
        # embed()
        prev_demo = demo 
        if not len(im_array) == 0:
            im_array[0].save('robo_frames/robot_traj_'+prev_demo+'.gif', save_all=True, append_images=im_array[1:], loop = 0)
            im_array = []


    file = open(image_fol+ '/new_crossing_count.txt', 'r')
    counter_crossing_data = file.read().split('\n')
    number_of_stops = len(counter_crossing_data)
    current_fol_number = int(image_fol.split('/')[-1])
    # print("Number of stops ", number_of_stops, counter_crossing_data)
    for counter_crossing in counter_crossing_data:
        counter_crossing = int(counter_crossing)
        if counter_crossing >= current_fol_number:
            break 
    feat_r[:,3,:] = get_traj_feature(feat_r[:,0], grid_size, human_past_traj)
    # traj_lower = full_trajs[:, 4].numpy()
    # traj_lower = traj_lower[~np.isnan(traj_lower).any(axis=1)]  # remove appended NAN rows
    # traj_lower = traj_lower.astype(np.int64)
    # traj_upper = full_trajs[:, 8].numpy()
    # traj_upper = traj_upper[~np.isnan(traj_upper).any(axis=1)]  # remove appended NAN rows
    # traj_upper = traj_upper.astype(np.int64)
    
    # feat_r[:,4,:] = get_traj_feature(feat_r[:,0], grid_size, robot_past_traj)
    # feat_h[:,4,:] = get_traj_feature(feat_h[:,0], grid_size, past_traj_h)
    # if not np.isnan(prev_predicted_traj_robot[start_full_index:end_full_index]).all():
    #     if not np.isnan(prev_past_traj_robot[start_full_index:end_full_index]).all():
    #         feat_h[:,5,:] = get_traj_feature(feat_h[:,0], grid_size, prev_past_traj_robot[start_full_index:end_full_index], prev_predicted_traj_robot[start_full_index:end_full_index])
    # feat_h[:,4,:] = get_traj_feature(feat_h[:,0], grid_size, past_traj_r, future_traj_r)
    print("Current folder is ", current_fol_number)
    print("Crossing counter is ", counter_crossing)
    tmp_nll_r, r_var_r, svf_diff_var_r, values_list_r, sampled_trajs_r, zeroing_loss_r = pred(feat_r, robot_traj, net, n_states, model, grid_size)
    for i in range(feat_r.shape[0]):
        # expected_return_lower = compute_return(r_var_r[i], traj_lower[i].type(torch.LongTensor))
        # expected_return_upper = compute_return(r_var_r[i], traj_upper[i].type(torch.LongTensor))
        expected_return_current = compute_return(r_var_r[i], np.array(sampled_trajs_r[i]))
        # returns_list.append([expected_return_lower, expected_return_upper, expected_return_current])
        # print("Expected return lower is ", expected_return_lower)
        # print("Expected return upper is ", expected_return_upper)
        # print("Expected return current is ", expected_return_current)
    r_vars_zeroed = r_var_r.clone()
    r_vars_zeroed = r_vars_zeroed*zeroing_loss_r

    c_zero = get_traj_length(robot_traj)/(grid_size*grid_size)
    # c_zero = np.zeros(c_zero.shape)
    grad_zeroed = torch.zeros(r_vars_zeroed.shape)
    for i in range(len(c_zero)):
        grad_zeroed[i] = c_zero[i]*(torch.ones(zeroing_loss_r[i].shape)-zeroing_loss_r[i])
    # traj_rank_weight = normalize_rank(demo_rank)
    c_zero = get_traj_length(robot_traj)/(grid_size*grid_size)
        # c_zero = np.zeros(c_zero.shape)
    for i in range(len(c_zero)):
        zeroing_loss_r[i] = c_zero[i]*zeroing_loss_r[i]
    zeroing_loss_criterion = zeroing_loss_r.mean()
    print("Demo rank is ", demo_rank)
    nll_test_list_robot += tmp_nll_r
    loss = [zeroing_loss_criterion]
    # if step % 1 == 0:
    visualize_counter = False
    for counter_crossing in counter_crossing_data:
        counter_crossing = int(counter_crossing)
        if abs(current_fol_number-counter_crossing)<5:
            visualize_counter = True
            break
    if visualize_counter:
        visualize_batch([robot_traj[0]], robot_traj, feat_r, r_var_r, values_list_r, zeroing_loss_r, current_fol_number, vis, grid_size, train=False, policy_sample_list=sampled_trajs_r, rank_list= demo_rank)
    vis.line(X=np.array([step]), Y=np.array([loss]), win=train_loss_win, update='append')
    print("Loss is ", loss) 
    step += 1
    traj_final = sampled_trajs_r[0]
    img = feat_r[:,0:3].numpy()
    img = img[0]
    data = img[0]
    for i in range(len(traj_final)):
        data[int(traj_final[i][0]), int(traj_final[i][1])] = 4.0

    make_plot_and_save(data, 'heatmap_temp.png')
    reward_data = r_var_r[:].detach().numpy()
    reward_data = reward_data[0][0]
    make_plot_and_save(reward_data, 'reward_temp.png')
# Open the saved image using PIL
    img = Image.open('heatmap_temp.png')
    reward_img = Image.open('reward_temp.png')
    full_img = get_concat_h(img, reward_img)
    # for i in range(len(traj_final)):
    #     img.putpixel((int(traj_final[i][0])*10, int(traj_final[i][1])*10), (255,0,0))
    # img.save('heatmap_temp_traj.png')
    im_array.append(full_img)
    # plt_frames.append([plt.imshow(data, cmap='hot', interpolation='nearest', animated=True)])

# ani = animation.ArtistAnimation(fig, plt_frames, interval=50, blit=True,
#                                 repeat_delay=1000)
im_array[0].save('robot_traj.gif', save_all=True, append_images=im_array[1:])
# ani.save("robot_traj.mp4")

nll_test_robot = sum(nll_test_list_robot) / len(nll_test_list_robot)
print('main. test nll {}'.format(nll_test_robot))
visualize_batch([robot_traj[0]], robot_traj, feat_r, r_var_r, values_list_r, r_vars_zeroed, step, vis, grid_size, train=False, policy_sample_list=sampled_trajs_r)

