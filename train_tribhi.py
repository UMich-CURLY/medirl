import mdp.offroad_grid as offroad_grid
from loader.data_loader_tribhi import OffroadLoader
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
import time
from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)

""" init param """
#pre_train_weight = 'pre-train-v6-dilated/step1580-loss0.0022763446904718876.pth'
pre_train_weight = None
vis_per_steps = 10
test_per_steps = 2
# resume = "step280-loss0.5675923794730127.pth"
resume = None
exp_name = '6.33'
grid_size = 32
discount = 0.9
lr = 5e-3
n_epoch = 64
batch_size = 16
n_worker = 8
use_gpu = True

if not os.path.exists(os.path.join('exp', exp_name)):
    os.makedirs(os.path.join('exp', exp_name))

host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://127.0.0.1', port=8098)
# vis = visdom.Visdom(env='main')

model = offroad_grid.OffroadGrid(grid_size, discount)
n_states = model.n_states
n_actions = model.n_actions

train_loader = OffroadLoader(grid_size=grid_size, tangent=False)
train_loader = DataLoader(train_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)
test_loader = OffroadLoader(grid_size=grid_size, train=False, tangent=False)
test_loader = DataLoader(test_loader, num_workers=n_worker, batch_size=batch_size, shuffle=True)

net = OnlyEnvDilated(feat_in_size = 4, feat_out_size = 50)
#net = OneStageDilated(feat_out_size=25)
step = 0
nll_cma = 0
nll_test = 0

if resume is None:
    if pre_train_weight is None:
        net.init_weights()
    else:
        pre_train_check = torch.load(os.path.join('exp', pre_train_weight))
        net.init_with_pre_train(pre_train_check)
else:
    checkpoint = torch.load(os.path.join('exp', exp_name, resume))
    step = checkpoint['step']
    net.load_state_dict(checkpoint['net_state'])
    nll_cma = checkpoint['nll_cma']
    # opt.load_state_dict(checkpoint['opt_state'])

opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

train_nll_win = vis.line(X=np.array([[-1, -1]]), Y=np.array([[nll_cma, nll_cma]]),
                         opts=dict(xlabel='steps', ylabel='loss', title='train acc'))
test_nll_win = vis.line(X=np.array([-1]), Y=np.array([nll_test]),
                        opts=dict(xlabel='steps', ylabel='loss', title='test acc'))

""" train """
best_test_nll = np.inf
for epoch in range(n_epoch):
    for _, (feat, traj) in enumerate(train_loader):
        start = time.time()
        net.train()
        print('main. step {}'.format(step))
        nll_list, r_var, svf_diff_var, values_list = pred(feat, traj, net, n_states, model, grid_size)
        
        opt.zero_grad()
        # embed()
        # a hack to enable backprop in pytorch with a vector
        # the normally used loss.backward() only works when loss is a scalar
        if not (np.isnan(values_list[0]).all().item()):
            torch.autograd.backward([r_var], [-svf_diff_var])  # to maximize, hence add minus sign
        else:
            embed()
        opt.step()
        nll = sum(nll_list) / len(nll_list)
        print('main. acc {}. took {} s'.format(nll, time.time() - start))
        print("Att epoch ", epoch)
        # cma. cumulative moving average. window size < 20
        nll_cma = (nll + nll_cma * min(step, 20)) / (min(step, 20) + 1)
        vis.line(X=np.array([[step, step]]), Y=np.array([[nll, nll_cma]]), win=train_nll_win, update='append')

        if step % vis_per_steps == 0 or nll > 10.5:
            visualize(traj, feat, r_var, values_list, svf_diff_var, step, vis, grid_size, train=True)
            if step == 0:
                step += 1
                continue

        if step % test_per_steps == 0:
            # test
            net.eval()
            nll_test_list = []
            for _, (feat, traj) in enumerate(test_loader):
                tmp_nll, r_var, svf_diff_var, values_list = pred(feat, traj, net, n_states, model, grid_size)
                nll_test_list += tmp_nll
            nll_test = sum(nll_test_list) / len(nll_test_list)
            print('main. test nll {}'.format(nll_test))
            vis.line(X=np.array([step]), Y=np.array([nll_test]), win=test_nll_win, update='append')
            # visualize(feat, r_variable, values, svf_diff_var, step, train=False)

            # if getting best test results, save weights
            if nll_test < best_test_nll:
                best_test_nll = nll_test
                state = {'nll_cma': nll_cma, 'test_nll': nll_test, 'step': step, 'net_state': net.state_dict(),
                         'opt_state': opt.state_dict(), 'discount':discount}
                path = os.path.join('exp', exp_name, 'step{}-loss{}.pth'.format(step, nll_test))
                torch.save(state, path)

        step += 1
