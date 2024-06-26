import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import warnings

warnings.filterwarnings('ignore')
from torch.autograd import Variable
import torch
from multiprocessing import Pool
from IPython import embed

def overlay_traj_to_map(traj, feat, value1=5.0):
    overlay_map = feat.copy()
    for i, p in enumerate(traj):
        overlay_map[int(p[0]), int(p[1])] = value1
    return overlay_map


def visualize_batch(past_traj, traj, feat, r_var, values, svf_diff_var, step, vis, grid_size, train=True, policy_sample_list = None):
    mode = 'train' if train else 'test'
    # n_batch = traj.shape[0]
    n_batch = 1
    for i in range(n_batch):
        # traj_sample = traj[i].numpy()  # choose one sample from the batch
        # traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        # traj_sample = traj_sample.astype(np.int64)

        vis.heatmap(X=feat[i, 0, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, step {} Goal Sink'.format(mode, step)))

        overlay_map = feat[i, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
        # overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
        
        # vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='{}, step {} semantic with traj self'.format(mode, step)))

        overlay_map = feat[i, 3, :, :].float().view(grid_size, -1).numpy()
        if policy_sample_list is not None:
            print(policy_sample_list[i])
            policy_sample = np.array(policy_sample_list[i])  # choose one sample from the batch

            policy_sample = policy_sample.astype(np.int64)
            overlay_map = overlay_traj_to_map(policy_sample, overlay_map)
        vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='{}, step {} semantic with learned traj'.format(mode, step)))
        if (feat.shape[1] >4):
            vis.heatmap(X=feat[i, 4, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='{}, Traj {} self'.format(mode, step)))
        if (feat.shape[1] >5):
            vis.heatmap(X=feat[i, 5, :, :].float().view(grid_size, -1),
                        opts=dict(colormap='Electric', title='{}, Traj {} other'.format(mode, step)))

        # vis.heatmap(X=feat[0, 3, :, :].float().view(grid_size, -1),
        #             opts=dict(colormap='Electric', title='{}, step {} Red Semantic'.format(mode, step)))
        
        vis.heatmap(X=r_var.data[i].view(grid_size, -1),
                    opts=dict(colormap='Greys', title='{}, step {}, rewards'.format(mode, step)))
        vis.heatmap(X=values[i].reshape(grid_size, -1),
                    opts=dict(colormap='Greys', title='{}, step {}, value'.format(mode, step)))
        if svf_diff_var is not None:
            vis.heatmap(X=svf_diff_var.data[i].view(grid_size, -1),
                        opts=dict(colormap='Greys', title='{}, step {}, SVF_diff'.format(mode, step)))


def visualize(past_traj, future_traj, feat, r_var, values, svf_diff_var, step, vis, grid_size, train=True, policy_sample_list = None):
    mode = 'train' if train else 'test'
    traj_sample = future_traj[0].numpy()  # choose one sample from the batch
    traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
    traj_sample = traj_sample.astype(np.int64)
    
    vis.heatmap(X=feat[0, 0, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, step {} Goal Sink'.format(mode, step)))

    overlay_map = feat[0, 1, :, :].float().view(grid_size, -1).numpy()  # (grid_size, grid_size)
    overlay_map = overlay_traj_to_map(traj_sample, overlay_map)
    
    vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='{}, step {} semantic with traj self'.format(mode, step)))

    overlay_map = feat[0, 3, :, :].float().view(grid_size, -1).numpy()
    if policy_sample_list is not None:
        print(policy_sample_list[0])
        policy_sample = np.array(policy_sample_list[0])  # choose one sample from the batch

        policy_sample = policy_sample.astype(np.int64)
        overlay_map = overlay_traj_to_map(policy_sample, overlay_map)
    vis.heatmap(X=overlay_map, opts=dict(colormap='Electric', title='{}, step {} semantic with learned traj'.format(mode, step)))
    vis.heatmap(X=feat[0, 4, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, Traj {} self'.format(mode, step)))
    vis.heatmap(X=feat[0, 5, :, :].float().view(grid_size, -1),
                opts=dict(colormap='Electric', title='{}, Traj {} other'.format(mode, step)))

    # vis.heatmap(X=feat[0, 3, :, :].float().view(grid_size, -1),
    #             opts=dict(colormap='Electric', title='{}, step {} Red Semantic'.format(mode, step)))
    
    vis.heatmap(X=r_var.data[0].view(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, rewards'.format(mode, step)))
    vis.heatmap(X=values[0].reshape(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, value'.format(mode, step)))
    vis.heatmap(X=svf_diff_var.data[0].view(grid_size, -1),
                opts=dict(colormap='Greys', title='{}, step {}, SVF_diff'.format(mode, step)))

    # for name, param in net.named_parameters():
    #    if name.endswith('weight'):
    #        vis.histogram(param.data.view(-1), opts=dict(numbins=20))  # weights
    #        vis.histogram(param.grad.data.view(-1), opts=dict(numbins=20))  # grads


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


### No multi processing for this one 
# def pred(feat, traj, net, n_states, model, grid_size):
#     n_sample = feat.shape[0]
#     feat = feat.float()
#     feat_var = Variable(feat)
#     r_var = net(feat_var)

#     result = []
#     for i in range(n_sample):
#         r_sample = r_var[i].data.numpy().squeeze().reshape(n_states)
#         traj_sample = traj[i].numpy()  # choose one sample from the batch
#         traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
#         traj_sample = traj_sample.astype(np.int64)
#         result.append(rl(traj_sample, r_sample, model, grid_size))

#     # extract result and stack svf_diff
#     # embed()
#     nll_list = [result[i].get()[0] for i in range(n_sample)]
#     svf_diff_var_list = [result[i].get()[1] for i in range(n_sample)]
#     values_list = [result[i].get()[2] for i in range(n_sample)]
#     svf_diff_var = torch.cat(svf_diff_var_list, dim=0)
#     return nll_list, r_var, svf_diff_var, values_list
