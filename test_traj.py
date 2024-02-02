import mdp.offroad_grid as offroad_grid
from loader.data_loader_test import OffroadLoader
from torch.utils.data import DataLoader
import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import visdom
import warnings
import logging
import os
import rospy
warnings.filterwarnings('ignore')
from network.hybrid_fcn import HybridFCN
from network.hybrid_dilated import HybridDilated
from network.one_stage_dilated import OneStageDilated
from network.only_env_dilated import OnlyEnvDilated
from network.reward_net import RewardNet

import torch
from torch.autograd import Variable
import time
from std_msgs.msg import Bool, Int32MultiArray, MultiArrayLayout, MultiArrayDimension

from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)
torch.set_default_tensor_type('torch.DoubleTensor')


resume = None
exp_name = '6.03robot'
grid_size = 32

resume = 'step120-loss1.6708339589222967.pth'

# host = os.environ['HOSTNAME']
# vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://127.0.0.1', port=8098)

def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj

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
            feat[i,int(x),int(y)] = 6
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

grid_size = 32

class irl():
    def __init__(self, grid_size=32):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("/Feature_publish/human_traj", Int32MultiArray, self.people_callback, queue_size=1)
        self.sub_robot = rospy.Subscriber("/Feature_publish/robot_traj", Int32MultiArray, self.robot_callback, queue_size=1)
        self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        discount = 0.9
        self._pub_traj = rospy.Publisher("irl_traj", Int32MultiArray, queue_size = 1)
        self.model = offroad_grid.OffroadGrid(grid_size, discount)
        self.n_states = self.model.n_states
        self.n_actions = self.model.n_actions
        n_worker = 1
        batch_size = 1
        loader = OffroadLoader(grid_size=grid_size, train=False)
        self.loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)
        self.net = RewardNet(n_channels=5, n_classes=1, n_kin = 0)
        # self.net.init_weights()
        checkpoint = torch.load(os.path.join('exp', exp_name, resume))
        self.net.load_state_dict(checkpoint['net_state'])

    def is_start(self, data):
        # self.net.eval()
        if not data.data:
            print ("Hasn't started yet")
            return False
        for step, (feat, robot_traj, human_traj) in enumerate(self.loader):
            start = time.time()
            human_traj = transpose_traj(np.array(self.human_traj))
            feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, [human_traj])
            feat_var = Variable(feat)
            r_var = self.net(feat)
            r_sample = r_var[0].data.numpy().squeeze().reshape(self.n_states)
            values_sample = self.model.find_optimal_value(r_sample, 0.1)
            policy = self.model.find_stochastic_policy(values_sample, r_sample)
            ### Can change to sampling longer trajectories
            sampled_traj = self.model.traj_sample(policy, grid_size, self.robot_traj[0][1], self.robot_traj[0][0])
            # nll_list, r_var, svf_diff_var, values_list, sampled_trajs_r, zeroing_loss_r = pred(feat, self.robot_traj, net, n_states, model, grid_size)
            # test_nll_list += nll_list
            # visualize_batch(self.robot_traj, [sampled_traj], feat, r_var, [values_sample], np.zeros([32,32]), step, vis, grid_size, train=False)
        traj = transpose_traj(np.array(sampled_traj))
        # traj = self.human_traj
        msg = Int32MultiArray()
        layout = MultiArrayLayout()
        dim0 = MultiArrayDimension()
        dim1 = MultiArrayDimension()
        dim0.label = "traj"
        dim1.label = "xy"
        dim0.stride = len(traj)*2
        dim0.size = len(traj)
        dim1.size = 2
        dim1.stride = 2
        layout.dim.append(dim0)
        layout.dim.append(dim1)
        layout.data_offset = 0
        msg.layout = layout
        data = np.array(traj)
        data = data.reshape([1,len(traj)*2])
        msg.data = np.ndarray.tolist(data[0])
        self._pub_traj.publish(msg)
        print(sampled_traj)
        rospy.sleep(5.0)
        # visualize_batch(np.array([self.robot_traj[0]]), None, feat, r_var, [values_sample], None , step, vis, grid_size, train=False, policy_sample_list=[sampled_traj])


    def people_callback(self, data):
        length = data.layout.dim[0].size
        self.human_traj = np.reshape(data.data, [length,2])

    def robot_callback(self,data):
        self.robot_traj = [data.data]

if __name__ == "__main__":
        rospy.init_node("Get_Traj",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = irl()
        update = 0
        while(not rospy.is_shutdown()):
            rospy.sleep(5.0)
            