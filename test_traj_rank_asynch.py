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
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, PointStamped, Point

from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)
torch.set_default_tensor_type('torch.DoubleTensor')
from message_filters import ApproximateTimeSynchronizer, Subscriber

resume = None
exp_name = '6.12robot'
resume  = 'step940-loss1.0095898266511534.pth'
GRID_RESOLUTION = 0.1
CLEARANCE_THRESH = 0.5/GRID_RESOLUTION
GRID_SIZE_IN_M = 6
grid_size = int(np.floor(GRID_SIZE_IN_M/GRID_RESOLUTION))
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://127.0.0.1', port=8099)

def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj

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
        traj_sample = traj[i]  # choose one sample from the batch
        traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        lengths.append(len(traj_sample))
    return np.array(lengths)

def get_traj_length_unique(traj):
    lengths = []
    traj_list_full = []
    for i in range(len(traj)):
        traj_sample = traj[i]  # choose one sample from the batch
        traj_sample = traj_sample[~np.isnan(traj_sample).any(axis=1)]  # remove appended NAN rows
        traj_list = []
        for j in range(len(traj_sample)):
            if list(traj_sample[j]) not in traj_list:
                traj_list.append([traj_sample[j][0], traj_sample[j][1]])
        lengths.append(len(traj_list))
        traj_list_full.append(traj_list)
    return np.array(lengths), traj_list_full

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

    
    return d

def get_data(*msgs):
    embed()

class irl():
    def __init__(self, grid_size=grid_size):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        self.sub_people = rospy.Subscriber("/Feature_expect/human_traj", Int32MultiArray, self.people_callback, queue_size=1)
        self.sub_robot = rospy.Subscriber("/Feature_expect/robot_traj", Int32MultiArray, self.robot_callback, queue_size=1)
        self.sub_img = rospy.Subscriber("/Feature_expect/rgb_grid", Int32MultiArray, self.img_callback, queue_size=1)
        # self.sub_people = Subscriber("/Feature_expect/human_traj", Int32MultiArray)
        # self.sub_robot_pos = Subscriber("/Feature_expect/robot_traj", Int32MultiArray)
        # self.sub_img = Subscriber("/Feature_expect/rgb_grid", Int32MultiArray)
        
        # self.ats = ApproximateTimeSynchronizer([self.sub_people, self.sub_robot_pos, self.sub_img], 10, 0.1, allow_headerless= True)
        
        # self.ats.registerCallback(self.get_data)
        # self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        self.sub_click = rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        discount = 0.9
        self._pub_traj = rospy.Publisher("irl_traj", Int32MultiArray, queue_size = 1)
        self._wait_traj = rospy.Publisher("wait_for_traj", Bool, queue_size = 1)
        self.model = offroad_grid.OffroadGrid(grid_size, discount)
        self.n_states = self.model.n_states
        self.n_actions = self.model.n_actions
        self.img = None 
        self.human_traj = []
        self.robot_past_traj = [] 
        self.robot_traj = None
        n_worker = 1
        batch_size = 1
        # loader = OffroadLoader(grid_size=grid_size, train=False)
        # self.loader = DataLoader(loader, num_workers=n_worker, batch_size=batch_size, shuffle=False)
        self.net = RewardNet(n_channels=4, n_classes=1, n_kin = 0)
        # self.net.init_weights()
        checkpoint = torch.load(os.path.join('exp', exp_name, resume))
        self.net.load_state_dict(checkpoint['net_state'])
        self.grid_size = grid_size
        self.counter = 0

    def is_start(self):
        # self.net.eval()
        # if not data.data:
        #     print ("Hasn't started yet")
        #     return False

        
        self._wait_traj.publish(True)

        if self.img is None:
            print("No image data found ")
            return 
        feat = torch.empty((1,4, grid_size, grid_size), dtype=torch.float64)
        # for step, (feat, robot_traj, human_past_traj, robot_past_traj)  in enumerate(self.loader):
        start = time.time()
        # human_traj = transpose_traj(np.array(self.human_traj))
        semantic_img_feat = np.array(self.img)[:,:,0:3].T
        
        for i in range(semantic_img_feat.shape[0]):
            # print(feat[i].shape)
            # print("Before nomalize", np.max(feat[i]))  
            semantic_img_feat[i] = semantic_img_feat[i] - np.mean(semantic_img_feat[i])*np.ones(semantic_img_feat[i].shape)
            if np.isclose(np.std(semantic_img_feat[i]), 0.0, atol = 0.0001):
                continue
            semantic_img_feat[i] = semantic_img_feat[i] / np.std(semantic_img_feat[i])*np.ones(semantic_img_feat[i].shape)
            if (np.isnan(semantic_img_feat[i]).any()):
                print("Still Nan somehow for feature ", i)
                semantic_img_feat[i] = semantic_img_feat[i] / np.mean(semantic_img_feat[i])*np.ones(semantic_img_feat[i].shape)
        feat[:,0:3,:] = torch.from_numpy(semantic_img_feat)
        feat[:,3,:] = get_traj_feature(feat[:,0], grid_size, [self.human_traj])
        # feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, [self.robot_past_traj])
        # feat_var = Variable(feat)
        r_var = self.net(feat)
        r_sample = r_var[0].data.numpy().squeeze().reshape(self.n_states)
        values_sample = self.model.find_optimal_value(r_sample, 0.1)
        policy = self.model.find_stochastic_policy(values_sample, r_sample)
        ### Can change to sampling longer trajectories

        print("Robot pos is ", self.robot_traj[0])
        sampled_traj = self.model.traj_sample(policy, grid_size, self.robot_traj[0], self.robot_traj[1])
        # nll_list, r_var, svf_diff_var, values_list, sampled_trajs_r, zeroing_loss_r = pred(feat, self.robot_traj, net, n_states, model, grid_size)
        # test_nll_list += nll_list
        # visualize_batch(self.robot_traj, [sampled_traj], feat, r_var, [values_sample], np.zeros([32,32]), step, vis, grid_size, train=False)
        traj = traj_interp(np.array(sampled_traj))
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
        print("Publishing traj msg", msg)
        visualize_batch(np.array([self.robot_traj[0]]), np.array([traj]), feat, r_var, [values_sample], None , self.counter, vis, grid_size, train=False, policy_sample_list=[traj])
        self._pub_traj.publish(msg)
        
        
        self.counter+=1

    def people_callback(self, data):
        length = data.layout.dim[0].size
        human_traj = np.reshape(data.data, [length,2])
        self.human_traj = []
        for i in range(length):
            loc = [human_traj[i][0], human_traj[i][1]]
            loc = self.contain_grid(loc)
            self.human_traj.append(loc)
        self.human_traj = np.array(self.human_traj)
        # self.human_traj = self.auto_pad_past(self.human_traj)

    def contain_grid(self, point):
        point[0] = max(point[0],0)
        point[0] = min(point[0], grid_size-1)
        point[1] = max(point[1],0)
        point[1] = min(point[1], grid_size-1)
        return point

    def robot_callback(self,data):
        length = data.layout.dim[0].size
        robot_past_traj = np.reshape(data.data, [length,2])
        self.robot_past_traj = []
        for i in range(length):
            loc = [robot_past_traj[i][0], robot_past_traj[i][1]]
            loc = self.contain_grid(loc)
            self.robot_past_traj.append(loc)
        # self.robot_past_traj = self.auto_pad_past(self.robot_past_traj)
        self.robot_past_traj = np.array(self.robot_past_traj)
        self.robot_traj = np.array([self.robot_past_traj[-1][0], self.robot_past_traj[-1][1]])

    def img_callback(self,data):
        self.img = np.reshape(data.data, [grid_size, grid_size, 3])
        print("Is empty? ", (self.img == np.zeros([grid_size, grid_size, 3])).all())
        self.is_start()
    
    def get_data(self, *msg):
        self._wait_traj.publish(True)
        print("In here length is ", len(msg))
        self.people_callback(msg[0])
        self.robot_callback(msg[1])
        self.img_callback(msg[2])
        print("On update ", msg[1])

    def point_callback(self, data):
        self.has_started = True

    def auto_pad_past(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size
        if traj.shape[0] >= self.grid_size:
            traj = traj[traj.shape[0]-self.grid_size:, :]
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size - traj.shape[0]
        pad_array = np.full((pad_len, 2), traj[0])
        output = np.vstack((traj, pad_array))
        return output
    
if __name__ == "__main__":
        rospy.init_node("Get_Traj",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = irl()
        update = 0
        while(not rospy.is_shutdown()):
            rospy.spin()
            