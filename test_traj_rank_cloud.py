#!/usr/bin/env python2.7

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
import sensor_msgs.point_cloud2 as pc2
import torch
from torch.autograd import Variable
import time
import threading
from PIL import Image
from std_msgs.msg import Bool, Int32MultiArray, MultiArrayLayout, MultiArrayDimension
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose, PointStamped, Point
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
import ctypes
import struct
from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)
torch.set_default_tensor_type('torch.DoubleTensor')
from message_filters import ApproximateTimeSynchronizer, Subscriber
import ros_numpy
resume = None
exp_name = '7.26robot'
resume  = 'step7200-loss1.3886675136775382.pth'
GRID_RESOLUTION = 0.1
CLEARANCE_THRESH = 0.5/GRID_RESOLUTION
GRID_SIZE_IN_M = 6
grid_size = int(np.floor(GRID_SIZE_IN_M/GRID_RESOLUTION))
TRAJ_LEN = 20
USE_VEL = True
VISUALIZE = True
if VISUALIZE:
    host = os.environ['HOSTNAME']
    vis = visdom.Visdom(env='v{}-{}'.format(exp_name, host), server='http://127.0.0.1', port=8099)
mutex = threading.Lock()
def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj

def get_heading_feat(pos, heading):
    temp_sink_feat = np.zeros([1,grid_size, grid_size])
    radius_in_px = [2,2]
    print("Position is ", (pos[0][0], pos[0][1]))
    robot_points = points_inside_circle((pos[0][0], pos[0][1]), radius_in_px[0])
    human_points = points_inside_circle((pos[1][0], pos[1][1]), radius_in_px[1])
    
    for point in robot_points:
        temp_sink_feat[0, point[0], point[1]] = heading[0] + np.pi + 1.0
    for point in human_points:
        temp_sink_feat[0, point[0], point[1]] = heading[1] + np.pi + 1.0
    return torch.from_numpy(temp_sink_feat)

def get_heading_human_feat(pos, heading):
    temp_sink_feat = np.zeros([1,grid_size, grid_size])
    radius_in_px = 2
    human_points = points_inside_circle((pos[0], pos[1]), radius_in_px)
    
    for point in human_points:
        temp_sink_feat[0, point[0], point[1]] = heading + np.pi + 1.0
    return torch.from_numpy(temp_sink_feat)

def get_vel_feat(pos, vel):
    temp_sink_feat = np.zeros([1,grid_size, grid_size])
    radius_in_px = 2
    human_points = points_inside_circle((pos[0], pos[1]), radius_in_px)
    
    for point in human_points:
        temp_sink_feat[0, point[0], point[1]] = vel * 3.0 + 1.0
    return torch.from_numpy(temp_sink_feat)

def get_goal_feat( goal_coords):
    temp_sink_feat = np.zeros([1,grid_size, grid_size])
    goal_points = points_inside_circle((goal_coords[0], goal_coords[1]), 2)

    for point in goal_points:
        temp_sink_feat[0, point[0], point[1]] = 6
    return torch.from_numpy(temp_sink_feat)

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
    img = np.zeros([grid_size, grid_size, 3])
    for i in range(goal_sink_feat.shape[0]):
        index = 1
        vals = np.linspace(0, 6, len(past_traj[i]))
        # print(past_traj[i].shape)
        for val in vals:
            [x,y] = past_traj[i][len(past_traj[i])-index]
            index = index+1
            if np.isnan([x,y]).any():
                continue
            feat[i,int(x),int(y)] = val
            img[int(x),int(y),:] = [int(val*255/6.0),0,0]
            
        if future_traj is not None:
            index = 0
            vals = np.linspace(3, 4 ,len(future_traj[i]))
            for val in vals:
                [x,y] = future_traj[i][index]
                if np.isnan([x,y]).any():
                    continue
                feat[i,int(x),int(y)] = val
                index = index+1
    
    # im=Image.fromarray(np.uint8(img))
    # im.save("feat.png")
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
def points_inside_circle( center, radius):
        cx =  int(center[0])
        cy =  int(center[1])
        points = []
        
        # Define the bounding box of the circle
        print("Center is ", center)
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                # Check if the point is inside the circle
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    if x >=0 and x <GRID_SIZE_IN_M/GRID_RESOLUTION and y>=0 and y<GRID_SIZE_IN_M/GRID_RESOLUTION:
                        points.append((x, y))
        
        return points

def auto_pad_past(traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = grid_size*50
        traj = traj_interp(traj)
        traj = np.array(traj)
        if traj.shape[0] >= fixed_len:
            traj = traj[0:fixed_len, :]
            traj = traj.astype(int)
            return traj
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = fixed_len - traj.shape[0]
        # pad_array = np.full((pad_len, 2), np.array([int(traj[-1][0]), int(traj[-1][1])]))
        pad_array = np.full((pad_len, 2), np.NaN)
        output = np.vstack((traj, pad_array))
        # print("Output is ", output)
        return output

class irl():
    def __init__(self, grid_size=grid_size):
        
        # self.traj_sub = rospy.Subscriber("traj_matrix", numpy_msg(Floats), self.traj_callback,queue_size=100)

        ### Replace with esfm
        # self.sub_people = rospy.Subscriber("/Feature_expect/human_traj", Int32MultiArray, self.people_callback, queue_size=1)
        # self.sub_robot = rospy.Subscriber("/Feature_expect/robot_traj", Int32MultiArray, self.robot_callback, queue_size=1)
        # self.sub_img = rospy.Subscriber("/Feature_expect/rgb_grid", Int32MultiArray, self.img_callback, queue_size=1)
        self.sub_people = Subscriber("/Feature_expect/human_path", Path)
        self.sub_robot_pos = Subscriber("/Feature_expect/robot_path", Path)
        self.sub_img = Subscriber("/semantic_cloud", PointCloud2)
        self.sub_robot_goal = rospy.Subscriber("/Feature_expect/robot_goal", Pose, self.goal_callback, queue_size=1)
        self.ats = ApproximateTimeSynchronizer([self.sub_people, self.sub_robot_pos], 5, 0.1, allow_headerless= False)
        
        self.ats.registerCallback(self.get_data)
        self.ats1 = ApproximateTimeSynchronizer([self.sub_robot_pos, self.sub_img], 5, 0.1, allow_headerless= False)
        self.ats1.registerCallback(self.get_data_2)
        # self.sub_ep_start = rospy.Subscriber("start_ep", Bool, self.is_start, queue_size=1)
        self.sub_click = rospy.Subscriber("/clicked_point", PointStamped,self.point_callback, queue_size=1)
        self.sub_ep_start = rospy.Subscriber("query_irl", Bool, self.is_start, queue_size=1)
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
        self.net = RewardNet(n_channels=7, n_classes=1, n_kin = 0)
        # self.net.init_weights()
        checkpoint = torch.load(os.path.join('exp', exp_name, resume))
        self.net.load_state_dict(checkpoint['net_state'])
        self.grid_size = grid_size
        self.counter = 0
        self.prev_reward = None
        self.prev_traj = None
        self.prev_robot_pose = None
        self.human_vel = 0.0

    def is_start(self, msg):
        # self.net.eval()
        # if not data.data:
        #     print ("Hasn't started yet")
        #     return False

        mutex.acquire(blocking=True)
        self._wait_traj.publish(True)

        if self.img is None:
            print("Caught ya")
            rospy.sleep(0.1)
            mutex.release()
            # embed()
            return 
        
        print("Made it past atleast once")
        feat = torch.empty((1,7, grid_size, grid_size), dtype=torch.float64)
        # for step, (feat, robot_traj, human_past_traj, robot_past_traj)  in enumerate(self.loader):
        start = time.time()
        # human_traj = transpose_traj(np.array(self.human_traj))
        semantic_img_feat = np.array(self.img)[:,:,0:3].T
        if (semantic_img_feat == np.zeros(semantic_img_feat.shape)).all():
            print("Image data is empty")
            mutex.release()
            return
        if self.human_traj is None:
            print("Human traj is empty")
            mutex.release()
            return
        if self.robot_traj is None:
            print("Robot traj is empty ")
            mutex.release()
            return
        if self.robot_goal is None:
            print("Robot goal is empty")
            mutex.release()
            return
        if self.robot_angle is None:    
            print("Robot angle is empty")
            mutex.release()
            return
        if self.human_angle is None:
            print("Human angle is empty")
            mutex.release()
            return
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
        print("Get heading of ", [self.robot_traj, self.human_traj[0]])
        # feat[:,4,:] = get_heading_feat([self.robot_traj, self.human_traj[0]], [self.robot_angle, self.human_angle])
        if feat.shape[1] == 5:
            feat[:,4,:] = get_goal_feat(self.robot_goal)
        else:
            feat[:,4,:] = get_heading_human_feat(self.human_traj[0], self.human_angle)
        if feat.shape[1] == 6:
            feat[:,5,:] = get_goal_feat(self.robot_goal)
        if feat.shape[1] == 7:
            feat[:,5,:] = get_vel_feat(self.human_traj[0], self.human_vel)
            feat[:,6,:] = get_goal_feat(self.robot_goal)
        # feat[:,4,:] = get_traj_feature(feat[:,0], grid_size, [self.robot_past_traj])
        # feat_var = Variable(feat)
        r_var = self.net(feat)
        r_sample = r_var[0].data.numpy().squeeze().reshape(self.n_states) 
        recalculate = True
        # if self.prev_reward is not None:
        #     if np.allclose(self.prev_reward, r_sample, atol=0.1) and (self.prev_robot_pose == self.robot_traj).all():
        #         recalculate = False
        if recalculate:
            self.prev_reward = r_sample.copy()   
            values_sample = self.model.find_optimal_value(r_sample, 0.1)
            
            policy = self.model.find_stochastic_policy(values_sample, r_sample)
            ### Can change to sampling longer trajectories

            # print("Robot pos is ", self.robot_traj[0])
            sampled_traj = self.model.traj_sample(policy, TRAJ_LEN, self.robot_traj[0], self.robot_traj[1])
            # nll_list, r_var, svf_diff_var, values_list, sampled_trajs_r, zeroing_loss_r = pred(feat, self.robot_traj, net, n_states, model, grid_size)
            # test_nll_list += nll_list
            
            # visualize_batch(self.robot_traj, [sampled_traj], feat, r_var, [values_sample], np.zeros([32,32]), step, vis, grid_size, train=False)
            traj = traj_interp(np.array(sampled_traj))
            self.prev_traj = traj
            self.prev_robot_pose = self.robot_traj.copy()
        else:
            print("Returning the same traj ")
            traj = self.prev_traj
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
        # print("Publishing traj msg", msg)
        self._pub_traj.publish(msg)
        self._wait_traj.publish(False)
        if VISUALIZE and recalculate:
            visualize_batch(np.array([self.robot_traj[0]]), torch.from_numpy(np.array([traj])), feat, r_var, [values_sample], None , self.counter, vis, grid_size, train=False, policy_sample_list=[traj])
        self.counter+=1
        self.img = None 
        self.robot_traj = None 
        self.human_traj = None

        mutex.release()

    

    def people_callback(self, msg):
        mutex.acquire(blocking=True)
        # print("Got new human pose", msg.poses)
        self.human_traj = []
        for pose in msg.poses:
            loc = [pose.pose.position.x/GRID_RESOLUTION, pose.pose.position.y/GRID_RESOLUTION]
            loc = self.contain_grid(loc)
            self.human_traj.insert(0,loc)
        self.human_traj = np.array(self.human_traj)
        # print("traj before padding is ", self.human_traj)
        self.human_traj = auto_pad_past(self.human_traj)
        self.human_traj.astype(int)
        # print("human traj is ", self.human_traj)
        orientation = [msg.poses[-1].pose.orientation.x, msg.poses[-1].pose.orientation.y, msg.poses[-1].pose.orientation.z, msg.poses[-1].pose.orientation.w]
        self.human_angle = msg.poses[-1].pose.orientation.z*10
        self.human_vel = msg.poses[-1].pose.position.z
        mutex.release()
        # self.human_traj = self.auto_pad_past(self.human_traj)

    def contain_grid(self, point):
        point[0] = int(np.round(max(point[0],0)))
        point[0] = int(np.round(min(point[0], grid_size-1)))
        point[1] = int(np.round(max(point[1],0)))
        point[1] = int(np.round(min(point[1], grid_size-1)))
        return point

    def robot_callback(self,msg):
        mutex.acquire(blocking=True)
        print("Got new robot_pose")
        self.robot_past_traj = []
        for pose in msg.poses:
            loc = [pose.pose.position.x/GRID_RESOLUTION, pose.pose.position.y/GRID_RESOLUTION]
            loc = self.contain_grid(loc)
            self.robot_past_traj.append(loc)
        # self.robot_past_traj = self.auto_pad_past(self.robot_past_traj)
        self.robot_past_traj = np.array(self.robot_past_traj)
        self.robot_traj = np.array([self.robot_past_traj[-1][0], self.robot_past_traj[-1][1]])
        orientation = [msg.poses[-1].pose.orientation.x, msg.poses[-1].pose.orientation.y, msg.poses[-1].pose.orientation.z, msg.poses[-1].pose.orientation.w]
        self.robot_angle = msg.poses[-1].pose.orientation.z*10
        mutex.release()

    def img_callback(self,msg):
        mutex.acquire(blocking=True)
        self.img = np.zeros([grid_size, grid_size,3])
        print("Got new img")
        pc=ros_numpy.numpify(msg)
        pc=ros_numpy.point_cloud2.split_rgb_field(pc)
        for pt_count in range(pc['x'].shape[0]):
            x,y = pc['x'][pt_count], pc['y'][pt_count]
            r,g,b = pc['r'][pt_count], pc['g'][pt_count], pc['b'][pt_count]
            self.img[int(np.round(y/GRID_RESOLUTION)), int(np.round(x/GRID_RESOLUTION)),:] = [r,g,b]
        mutex.release()
        # self.is_start()
    
    def goal_callback(self, msg):
        mutex.acquire(blocking=True)
        print("Got new goal")
        self.robot_goal = [msg.position.x/GRID_RESOLUTION, msg.position.y/GRID_RESOLUTION]
        self.robot_goal = self.contain_grid(self.robot_goal)
        mutex.release()

    def get_data(self, *msg):
        
        print("In here length is ", len(msg))
        self.people_callback(msg[0])
        self.robot_callback(msg[1])
        # self.img_callback(msg[2])
        

    def get_data_2(self, *msg):
        self.robot_callback(msg[0])
        self.img_callback(msg[1])
        # print("On update ", msg[1])

    def point_callback(self, data):
        self.has_started = True



    
if __name__ == "__main__":
        rospy.init_node("Get_Traj",anonymous=False)
        # initpose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
        feature = irl()
        update = 0
        while(not rospy.is_shutdown()):
            rospy.spin()
            