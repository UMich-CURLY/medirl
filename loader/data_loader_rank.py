#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from torch.utils.data import Dataset
import os
import scipy.io as sio
#from skimage.transform import resize
#from skimage.transform import rotate
from PIL import Image
import numpy as np
from scipy import optimize
from IPython import embed
np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math
import csv 
USE_GOAL = True
FIXED_LEN = 20
USE_VEL = True
USE_HEADING = True
def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj

def check_neighbor(first, second):
    actions = np.array([[0,0], [1,0], [-1,0], [0,1], [0,-1]])
    possible_neighbors = list(first) + actions
    for neighbor in possible_neighbors:
        if (second==neighbor).all():
            return True
    return False
    

def remove_oscilations(traj):
    already_done = []
    current_targ_list = []
    for i in range(len(traj)-2):
        if i in already_done:
            continue
        if (traj[i] == traj[i+2]).all():
            for j in range(i, len(traj)):
                if (traj[j] == traj[i]).all():
                    current_targ_list.append(j)
            # print("Current target list is ", current_targ_list)
            # print("Current trag contender is ", traj[i])
            if len(current_targ_list) > 2:
                traj[current_targ_list[0]:current_targ_list[-1]] = traj[current_targ_list[0]]
                already_done.append(current_targ_list)
                current_targ_list = []
    # print("TYraj after removing osciallations is ", traj)
    return traj

def is_valid_traj(traj):
    i = 0
    while i < len(traj)-1:
        if not check_neighbor(traj[i], traj[i+1]):
            return False
        i = i+1
    return True
def get_traj_length_unique_with_osc(traj):
    lengths = []
    traj_list = []
    for j in range(len(traj)):
        # if list(traj[j]) not in traj_list:
        if j>0:
            if (traj[j] == traj[j-1]).all():
                continue
        traj_list.append([traj[j][0], traj[j][1]])
    lengths.append(len(traj_list))
    return np.array(lengths), np.array(traj_list)

def get_traj_length_unique_actual(traj):
    lengths = []
    traj_list = []
    traj = remove_oscilations(traj)
    for j in range(len(traj)):
        # if list(traj[j]) in traj_list:
        #     continue
        if j>0:
            if (traj[j] == traj[j-1]).all():
                continue
        # if True:
        traj_list.append([traj[j][0], traj[j][1]])
    lengths.append(len(traj_list))
    return np.array(lengths), np.array(traj_list)

def traj_interp(c):
    d = c.astype(int)
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
    # connected_map = np.zeros((32, 32))
    # for i in range(len(d)):
    #     connected_map[int(d[i,1])+1, int(d[i,0])+1] = 1
    if not is_valid_traj(d):
        print(d)
    # print(c)
    # print(d)
    # print("Valid traj? ", is_valid_traj(d), d.shape, c.shape)
    return np.array(d)

class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='data/irl_sept_24_3', pre_train=False, tangent=False,
                 more_kinematic=None, human = False):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        self.data_dir = datadir
        self.training = train
        if train:
            self.data_dir = datadir + '/train'
        else:
            self.data_dir = datadir + '/test'
        # print(self.data_dir)
        # if not train:
        #     self.data_dir = '/root/medirl/data/irl_driving_full_future'
        # if demo is not None:
        #     self.data_dir = datadir + '/irl_data/' + demo
        demos =  os.listdir(self.data_dir)
        try:
            demos.remove('metrics_data.csv')
        except:
            pass
        self.metrics = []
        with open(self.data_dir+"/metrics_data.csv", mode='r', newline='') as file:
            # Create a DictReader object
            csv_reader = csv.DictReader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Print each row as a dictionary
                print(row)
                self.metrics.append(row)
        self.ep_list = {}
        for metric in self.metrics:
            if metric['ep_no'] in self.ep_list.keys():
                # try:
                new_list = self.ep_list[metric['ep_no']].copy()
                new_list.append(int(metric['reset_counter']))
                # self.ep_list.update({metric['ep_no']: [self.ep_list[metric['ep_no']][0:-1][0], (int(metric['reset_counter']))]})
                self.ep_list.update({metric['ep_no']: new_list})
                # except:
                
            else:
                self.ep_list.update({metric['ep_no']: [int(metric['reset_counter'])]})
                
        demos.sort(key=lambda x:int(x[5:]))
        self.data_list = []
        remove_list = ['traj.npy']
        for demo in demos:
            # if demo == "demo_22" or demo == "demo_25" or demo == "demo_28":
            #     continue
            items = os.listdir(self.data_dir+"/"+demo)
            items = [ x for x in items if x.isdigit() ]
            items.sort(key=lambda x:int(x))
            with open(self.data_dir+"/"+demo + "/traj.npy", 'rb') as f:
                full_traj = np.load(f)
            full_traj = remove_oscilations(full_traj)

            with open(self.data_dir+"/"+demo + "/new_traj.npy", 'wb') as f:
                np.save(f, np.array(full_traj))
            try:
                with open(self.data_dir+"/"+demo+"/tokeep.npy", 'rb') as f:
                    items_to_keep = np.load(f)
            except:
                items_to_keep = []
            for item in items:
                if int(item) not in items_to_keep and items_to_keep != []:
                    continue
                file = open(self.data_dir+"/"+demo + '/' + "new_crossing_count.txt", 'r')
                counter_crossing_data = file.read().split('\n')
                keep_it = False
                for crossing in counter_crossing_data:
                    if int(item) <= int(crossing)+5 and int(item) >= int(crossing)-5:
                        weight = 1.0
                        with open(self.data_dir+"/"+demo + '/' + item + '/weight.txt', 'w') as f:
                            f.write(str(weight))
                            f.close()
                        keep_it = True
                    else:
                        weight = 0.2
                        with open(self.data_dir+"/"+demo + '/' + item + '/weight.txt', 'w') as f:
                            f.write(str(weight))
                            f.close()
                if (int(item) %2 != 0 and not keep_it) or int(item) > int(items[-1])-4:
                    continue
                if int(item) == 0:
                    continue
                
                robot_past_traj = self.data_dir+"/"+demo + '/' + item+"/robot_past_traj.npy"
                # if item in remove_list:
                #     continue
                if not (os.path.exists(robot_past_traj)):
                    continue
                rank_path = self.data_dir+"/"+demo + '/new_rank.txt'
                if not (os.path.exists(rank_path)):
                    continue
                file = open(self.data_dir+'/'+demo + '/new_rank.txt', 'r')
                demo_rank = float(file.read())
                if demo_rank <=0.99: ### used to be <=0.2 normally, training only for optimal episodes
                    continue
                # if (self.check_isnone(self.data_dir + '/' + item)):
                #     continue
                human_past_traj = self.data_dir+"/"+demo + '/' + item+"/human_past_traj.npy"
                if not (os.path.exists(human_past_traj)):
                    continue
                grid_img = self.data_dir+"/"+demo + '/' + item+"/grid_map.png"
                if not (os.path.exists(grid_img)):
                    continue
                


                # if (self.check_isnone(self.data_dir + '/' + item)):
                #     continue
                __ = os.system("cp "+self.data_dir+"/"+demo + '/' + "new_traj.npy " + self.data_dir+"/"+demo + '/' + item)
                __ = os.system("cp "+self.data_dir+"/"+demo + '/' + "new_rank.txt " + self.data_dir+"/"+demo + '/' + item)
                __ = os.system("cp "+self.data_dir+"/"+demo + '/' + "crossing_count.txt " + self.data_dir+"/"+demo + '/' + item)
                __ = os.system("cp "+self.data_dir+"/"+demo + '/' + "new_crossing_count.txt " + self.data_dir+"/"+demo + '/' + item)
                
                self.data_list.append(self.data_dir+"/"+demo + '/' + item)

        print(self.data_list)
        #self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train
        # kinematic related feature
        self.center_idx = self.grid_size / 2
    
    def points_inside_circle(self, center, radius):
        cx, cy = center
        points = []
        
        # Define the bounding box of the circle
        for x in range(cx - radius, cx + radius + 1):
            for y in range(cy - radius, cy + radius + 1):
                # Check if the point is inside the circle
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    if x >=0 and x <self.grid_size:
                        if y>=0 and y<self.grid_size:
                            points.append((x, y))
        
        return points

    def check_isnone(self,image_fol):
        self.image_fol = image_fol
        goal_sink_feat = np.array(Image.open(self.image_fol+"/goal_sink.png")).T
        goal_sink_feat = goal_sink_feat/255*np.ones(goal_sink_feat.shape)
        temp_sink_feat = np.zeros([1,goal_sink_feat.shape[1], goal_sink_feat.shape[2]])
        for i in range(goal_sink_feat.shape[1]):
            for j in range(goal_sink_feat.shape[2]):
                if (goal_sink_feat[1,i,j] == 1):
                    temp_sink_feat[0,i,j] = 5
                if (goal_sink_feat[2,i,j] == 1):
                    temp_sink_feat[0,i,j] = 10
        goal_sink_feat = temp_sink_feat
        semantic_img_feat = np.array(Image.open(self.image_fol+"/semantic_img.png"))[:,:,0:3].T
        with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
            traj = np.load(f)
            if (len(traj) == 0):
                return True

        for i in range(traj.shape[0]):
            temp = traj[i,0] 
            traj[i,0]= traj[i,1]
            traj[i,1] = temp 
        # visualize rgb
        # with open(self.image_fol+"/sdf.npy", 'rb') as f:
        #     sdf_feat = np.load(f)
        feat = np.concatenate((goal_sink_feat, semantic_img_feat), axis = 0)
        # normalize features locally

        for i in range(feat.shape[0]):
            # print(feat[i].shape)
            # print("Before nomalize", np.max(feat[i]))  
            feat[i] = feat[i] - np.mean(feat[i])*np.ones(feat[i].shape)
            feat[i] = feat[i] / np.std(feat[i])*np.ones(feat[i].shape)
            if (np.isnan(feat[i]).any()):
                return True
        return False
    
    



    def get_heading_feat(self,pos, heading):
        temp_sink_feat = np.zeros([1,self.grid_size, self.grid_size])
        radius_in_px = [2,2]
        # print("Position is ", (pos[0][0], pos[0][1]))
        robot_points = self.points_inside_circle((pos[0][0], pos[0][1]), radius_in_px[0])
        human_points = self.points_inside_circle((pos[1][0], pos[1][1]), radius_in_px[1])
        
        for point in robot_points:
            temp_sink_feat[0, point[0], point[1]] = heading[0] + np.pi + 1.0
        for point in human_points:
            temp_sink_feat[0, point[0], point[1]] = heading[1] + np.pi + 1.0
        return temp_sink_feat

    def get_heading_feat_human(self,pos, heading):
        temp_sink_feat = np.zeros([1,self.grid_size, self.grid_size])
        radius_in_px = 2
        human_points = self.points_inside_circle((pos[0], pos[1]), radius_in_px)
        
        for point in human_points:
            temp_sink_feat[0, point[0], point[1]] = heading[1] + np.pi + 1.0
        return temp_sink_feat

    def get_vel_feat(self,pos, vel):
        temp_sink_feat = np.zeros([1,self.grid_size, self.grid_size])
        radius_in_px = 2
        human_points = self.points_inside_circle((pos[0], pos[1]), radius_in_px)
        
        for point in human_points:
            temp_sink_feat[0, point[0], point[1]] = vel * 3.0 + 1.0
        return temp_sink_feat

    def get_goal_feat(self, goal_coords):
        temp_sink_feat = np.zeros([1,self.grid_size, self.grid_size])
        goal_points = self.points_inside_circle((goal_coords[0], goal_coords[1]), 2)

        for point in goal_points:
            temp_sink_feat[0, point[0], point[1]] = 6
        return temp_sink_feat


    def __getitem__(self, item_num):
        self.image_fol = self.data_list[item_num]
        # print(self.image_fol)
        file = open(self.image_fol+ '/new_rank.txt', 'r')
        demo_rank = float(file.read())
        try:
            if self.training:
                weight_file = open(self.image_fol+ '/weight.txt', 'r')
                weight = float(weight_file.read())
            else:
                weight = 0.2
        except:
            weight = 0.2

        file = open(self.image_fol+ '/new_crossing_count.txt', 'r')
        counter_crossing_data = file.read().split('\n')
        number_of_stops = len(counter_crossing_data)
        current_fol_number = int(self.image_fol.split('/')[-1])
        # print("Number of stops ", number_of_stops, counter_crossing_data)
        for counter_crossing in counter_crossing_data:
            counter_crossing = int(counter_crossing)
            if counter_crossing >= current_fol_number:
                break
        print("On folder ", self.image_fol)
        # goal_sink_feat = np.array(Image.open(self.image_fol+"/goal_sink.png")).T
        # goal_sink_feat = goal_sink_feat/255*np.ones(goal_sink_feat.shape)
        # temp_sink_feat = np.zeros([1,goal_sink_feat.shape[1], goal_sink_feat.shape[2]])
        # for i in range(goal_sink_feat.shape[1]):
        #     for j in range(goal_sink_feat.shape[2]):
        #         if (goal_sink_feat[1,i,j] == 1):
        #             temp_sink_feat[0,i,j] = 5
        #         if (goal_sink_feat[2,i,j] == 1):
        #             temp_sink_feat[0,i,j] = 10
        # goal_sink_feat = temp_sink_feat
        semantic_img_feat = np.array(Image.open(self.image_fol+"/grid_map.png"))[:,:,0:3].T
        
        # robot_traj = transpose_traj(robot_traj)
        # if not is_valid_traj(past_traj) or not is_valid_traj(past_other_traj):
        #     print("Bad past traj in ", self.image_fol)
            # embed()
        with open(self.image_fol+"/heading.npy", 'rb') as f:
            heading_angle = np.load(f)
        # if len(full_traj) == 0:
        #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
        #         full_traj = np.load(f)
        
        with open(self.image_fol+"/human_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        # if len(full_traj) == 0:
        #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
        #         full_traj = np.load(f)
        full_traj = traj_interp(full_traj)
        human_past_traj = remove_oscilations(full_traj)
        # length, human_past_traj = get_traj_length_unique_actual(full_traj)
        human_past_traj = traj_interp(human_past_traj)

        with open(self.image_fol+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        # if len(full_traj) == 0:
        #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
        #         full_traj = np.load(f)
        full_traj = traj_interp(full_traj)
        robot_past_traj = remove_oscilations(full_traj)
        # length, robot_past_traj = get_traj_length_unique_actual(full_traj)
        robot_past_traj = traj_interp(robot_past_traj)

        with open(self.image_fol+"/new_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        full_traj = np.array(traj_interp(full_traj), np.int)
        robot_traj = remove_oscilations(full_traj)
        # print("Valid full traj? ", is_valid_traj(full_traj))
        # length, robot_traj = get_traj_length_unique_actual(full_traj)
        robot_traj = traj_interp(robot_traj)
        # past_ind= np.where(robot_traj == robot_past_traj[-1])
        # print("Fial traj ", is_valid_traj(robot_traj))
        # if (not is_valid_traj(robot_traj)):
        #     print(past_ind)
        #     print(robot_past_traj[-1], robot_traj, len)
        #     print(temp)
        # robot_past_traj = transpose_traj(robot_past_traj)
        robot_pos = np.array([robot_past_traj[-1,0], robot_past_traj[-1,1]])
        human_pos = np.array([human_past_traj[-1,0], human_past_traj[-1,1]])
        # heading_feat = self.get_heading_feat([robot_pos, human_pos], heading_angle)
        heading_feat = self.get_heading_feat_human([human_pos[0], human_pos[1]], heading_angle)
        goal_feat = None
        if os.path.exists(self.image_fol+"/goal.npy"):
            with open(self.image_fol+ "/goal.npy", 'rb') as f:
                goal_coords = np.load(f)
            goal_feat = self.get_goal_feat(goal_coords)
        if USE_VEL is True:
            with open(self.image_fol+"/vels.npy", 'rb') as f:
                vels = np.load(f)
            human_vel = vels[1] 
            vel_feat = self.get_vel_feat([human_pos[0], human_pos[1]], human_vel)
            heading_feat = np.concatenate((heading_feat, vel_feat), axis = 0)
        # if len(human_past_traj) == 0:
        #     human_past_traj = np.array([human_future_traj[0]])
        # human_past_traj = np.append(human_past_traj, np.array([human_future_traj[0]]))
        # if not is_valid_traj(future_traj) or not is_valid_traj(future_other_traj):
        #     print("Bad future traj in ", self.image_fol)
            # embed()
        # if past_traj_len == 0:
        #     human_past_traj = np.array([[human_future_traj[0,0], human_future_traj[0,1]]])
        # visualize rgb
        # with open(self.image_fol+"/sdf.npy", 'rb') as f:
        #     sdf_feat = np.load(f)
        # feat = np.concatenate((goal_sink_feat, semantic_img_feat), axis = 0)
        ### Add the traj features 
        robot_traj = self.auto_pad_future_from_past_counter(robot_traj[:, :2], robot_past_traj, counter_crossing)
        # human_past_traj = self.auto_pad_past(human_past_traj[:, :2]).T
        # robot_past_traj = self.auto_pad_past(robot_past_traj[:,:2]).T
        # past_ind = 0
        # for i in range(len[0]):
        #     if (robot_traj[i] == robot_past_traj[-1]).all():
        #         past_ind = i
        # temp = robot_traj
        # robot_traj = robot_traj[past_ind:]
        
        # len = robot_traj.shape[0]
        # if len == 1:
        #     robot_traj = np.vstack((robot_traj, np.array([robot_traj[0,0]-1, robot_traj[0,1]])))
        
        human_traj_feature = np.zeros([1, semantic_img_feat.shape[1], semantic_img_feat.shape[2]])
        human_traj_feature[0,list(np.array(human_past_traj[:,0], np.int)), list(np.array(human_past_traj[:,1], np.int))] = 100
        # other_traj_feature = np.zeros(goal_sink_feat.shape)
        robot_traj_feature = np.zeros([1,semantic_img_feat.shape[1], semantic_img_feat.shape[2]])
        # robot_traj_feature[0,list(np.array(robot_past_traj[:,0], np.int)), list(np.array(robot_past_traj[:,1], np.int))] = 100
        # other_traj_feature[0,list(np.array(past_other_traj[:,0], np.int)), list(np.array(past_other_traj[:,1], np.int))] = 100
        # kin_feats = np.concatenate((self_traj_feature, ther_traj_feature), axis = 0)
        feat = np.concatenate((semantic_img_feat, human_traj_feature), axis = 0)
        # feat = np.concatenate((feat, robot_traj_feature), axis = 0)
        # normalize features locally

        for i in range(feat.shape[0]):
            # print(feat[i].shape)
            # print("Before nomalize", np.max(feat[i]))  
            feat[i] = feat[i] - np.mean(feat[i])*np.ones(feat[i].shape)
            if np.isclose(np.std(feat[i]), 0.0, atol = 0.0001):
                continue
            feat[i] = feat[i] / np.std(feat[i])*np.ones(feat[i].shape)
            if (np.isnan(feat[i]).any()):
                print("Still Nan somehow for feature ", i)
                feat[i] = feat[i] / np.mean(feat[i])*np.ones(feat[i].shape)
            # print("for i mean is std is ", i, np.mean(feat[i]), np.std(feat[i]))
            # print("After normalize min max", np.min(feat[i]), np.max(feat[i]))   
        if USE_HEADING:    
            feat = np.concatenate((feat, heading_feat), axis = 0)
        if goal_feat is not None and USE_GOAL is True:
            feat = np.concatenate((feat, goal_feat), axis = 0)
        
        current_fol_number = int(self.image_fol.split('/')[-1])
        demo_num = int(self.image_fol.split('/')[-2].split('_')[1])
        full_traj_array = np.NaN*np.ones((9,20,2))
        if "train" in self.image_fol:
            for ep in self.ep_list:
                if demo_num in self.ep_list[ep]:
                    break
            demos = self.ep_list[ep]
            for demo in demos:
                with open(self.data_dir+'/demo_'+str(demo)+"/new_traj.npy", 'rb') as f:
                    full_traj = np.load(f)
                full_traj = np.array(traj_interp(full_traj), np.int)
                robot_traj_full = full_traj
                # length, robot_traj_full = get_traj_length_unique_actual(full_traj)
                robot_traj_full = traj_interp(robot_traj_full)
                file = open(self.data_dir+'/demo_'+str(demo)+ '/new_crossing_count.txt', 'r')
                new_counter_crossing_data = file.read().split('\n')
                
                items = os.listdir(self.data_dir+"/demo_"+str(demo))
                items = list([ int(x) for x in items if x.isdigit() ])
                if current_fol_number >= np.max(items):
                    current_fol_number_now = np.max(items)
                else:
                    current_fol_number_now = current_fol_number
                # print("Number of stops ", number_of_stops, counter_crossing_data)
                for new_counter_crossing in new_counter_crossing_data:
                    new_counter_crossing = int(new_counter_crossing)
                    if new_counter_crossing >= current_fol_number_now:
                        break
                # print("New Counter crossing ", new_counter_crossing)
                # new_counter_crossing = int(file.read())
                # print("Current fol number now is and crossing count is ", '/demo_'+str(demo)+ '/'+str(current_fol_number_now), new_counter_crossing)
                with open(self.data_dir+'/demo_'+str(demo)+ '/'+str(current_fol_number_now) + '/robot_past_traj.npy', 'rb') as f:
                    full_traj = np.load(f)
                full_traj = traj_interp(full_traj)
                robot_past_traj_now = full_traj
                # length, robot_past_traj_now = get_traj_length_unique_actual(full_traj)
                robot_past_traj_now = traj_interp(robot_past_traj_now)
                # robot_traj_full = self.auto_pad_future_from_past_other(robot_traj_full[:, :2], robot_pos, new_counter_crossing, self.data_dir+'/demo_'+str(demo)+ '/'+str(current_fol_number))
                robot_traj_full = self.auto_pad_future_from_past_counter(np.array(robot_traj_full[:, :2], dtype=np.float32), robot_past_traj_now, new_counter_crossing, self.data_dir+'/demo_'+str(demo)+ '/'+str(current_fol_number_now))
                # print("Robot traj full is", robot_traj_full)
                file = open(self.data_dir+'/demo_'+str(demo)+ '/new_rank.txt', 'r')
                demo_rank_full = float(file.read())
                if demo_rank_full <=0.2:
                    continue
                one_hot_rank = int(demo_rank_full*10)
                full_traj_array[one_hot_rank-2] = robot_traj_full
        human_past_traj = self.auto_pad_past(human_past_traj[:, :2]).T
        robot_past_traj = self.auto_pad_past(robot_past_traj[:,:2]).T
        # future_other_traj = self.auto_pad_future(future_other_traj[:, :2])
        return feat, robot_traj, human_past_traj, robot_past_traj, demo_rank, weight, full_traj_array

    def __len__(self):
        return len(self.data_list)

    def auto_pad_past(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size*100
        if traj.shape[0] >= fixed_len:
            traj = traj[traj.shape[0]-fixed_len:, :]
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = fixed_len - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.NaN)
        output = np.vstack((pad_array, traj))
        return output


    def auto_pad_future_from_past_other(self, traj, robot_pos, counter, counter_fol):
        fixed_len = FIXED_LEN
        # print("Traj shape is ", traj.shape)
        # for i in range(traj.shape[0]):
        norm_list = np.linalg.norm(traj-robot_pos*np.ones(traj.shape), axis = 1)
        # print("Norm list is ", norm_list)
        past_len = np.argmin(norm_list)
        # past_len = past_traj.shape[0]
        print("Past len is ", past_len)
        if counter_fol is None:
            counter_fol = self.data_dir+"/"+self.image_fol.split('/')[-2] + '/' + str(counter)
        else:
            counter_fol = self.data_dir+"/"+counter_fol.split('/')[-2] + '/' + str(counter)
            
        counter_fol_past_traj = np.load(counter_fol+"/robot_past_traj.npy")
        counter_fol_past_traj = traj_interp(counter_fol_past_traj)
        lengh, counter_fol_past_traj = get_traj_length_unique_actual(counter_fol_past_traj)
        counter_fol_past_traj = traj_interp(counter_fol_past_traj)
        counter_fol_past_len = counter_fol_past_traj.shape[0]
        # print("Counter past len is ", counter_fol_past_len)
        current_fol_number = int(self.image_fol.split('/')[-1])
        # print("Folder splits are ",self.image_fol.split('/'))
        # print("Current fol number is ", current_fol_number)
        if past_len<counter_fol_past_len:
            traj = counter_fol_past_traj
            print("Before the counter", counter)
        if traj.shape[0]-past_len<fixed_len:
            if past_len<traj.shape[0]:
                traj = traj[past_len-1:,:]
            else:
                traj = traj[-1:,:]
            # traj = traj.astype(int)
            length_unique, traj = get_traj_length_unique_actual(traj)
            traj = traj_interp(traj)
            pad_len = fixed_len - traj.shape[0]    
            pad_list = []
            for i in range(int(np.ceil(pad_len))):
                if (i < pad_len):
                    pad_list.append([traj[-1,0], traj[-1,1]])
                else:
                    pad_list.append([np.NaN, np.NaN])
            pad_array = np.array(pad_list[:pad_len])
            # traj = traj.astype(int)
            if pad_len>0:
                output = np.vstack((traj, pad_array))
            else:
                output = traj
            return  output
        traj = traj[past_len:past_len+fixed_len,:]
        # traj = traj.astype(int)
        length_unique, traj = get_traj_length_unique_actual(traj)
        traj = traj_interp(traj)
        pad_len = fixed_len - traj.shape[0]    
        pad_list = []
        for i in range(int(np.ceil(pad_len))):
            if (i < pad_len):
                pad_list.append([traj[-1,0], traj[-1,1]])
            else:
                pad_list.append([np.NaN, np.NaN])
        pad_array = np.array(pad_list[:pad_len])
        # traj = traj.astype(int)
        if pad_len>0:
            output = np.vstack((traj, pad_array))
        else:
            output = traj
        return  output
        
        if not is_valid_traj(traj):
            print("Invalid traj ", traj, current_fol_number, counter)
            # embed()
        return traj

    def auto_pad_future_from_past(self, traj, past_traj):
        fixed_len = FIXED_LEN
        past_len = past_traj.shape[0]
        print("Past len is ", past_len)
        
        if traj.shape[0]-past_len<fixed_len:
            if past_len<traj.shape[0]:
                traj = traj[past_len:,:]
            else:
                traj = traj[-1:,:]
            traj = traj.astype(int)
            pad_len = fixed_len - traj.shape[0]    
            pad_list = []
            for i in range(int(np.ceil(pad_len))):
                if (i < pad_len):
                    pad_list.append([traj[-1,0], traj[-1,1]])
                else:
                    pad_list.append([np.NaN, np.NaN])
            pad_array = np.array(pad_list[:pad_len])
            if pad_len>0:
                output = np.vstack((traj, pad_array))
            else:
                output = traj
            return  output
        traj = traj[past_len:past_len+fixed_len,:]
        traj = traj.astype(int)
        return traj

    def auto_pad_future_from_past_counter(self, traj, past_traj, counter, counter_fol = None):
        fixed_len = FIXED_LEN
        past_len = past_traj.shape[0]
        # print("Past len is ", past_len)
        if counter_fol is None:
            counter_fol = self.data_dir+"/"+self.image_fol.split('/')[-2] + '/' + str(counter)
        else:
            counter_fol = self.data_dir+"/"+counter_fol.split('/')[-2] + '/' + str(counter)
        # print("Counter fol is ", counter_fol)
        counter_fol_past_traj = np.load(counter_fol+"/robot_past_traj.npy")
        counter_fol_past_traj = traj_interp(counter_fol_past_traj)
        # lengh, counter_fol_past_traj = get_traj_length_unique_actual(counter_fol_past_traj)
        counter_fol_past_traj = remove_oscilations(counter_fol_past_traj)
        counter_fol_past_traj = traj_interp(counter_fol_past_traj)
        counter_fol_past_len = counter_fol_past_traj.shape[0]
        # print("Counter past len is ", counter_fol_past_len)
        if counter_fol is None:
            current_fol_number = int(self.image_fol.split('/')[-1])
        else:
            current_fol_number = int(counter_fol.split('/')[-1])

        # print("Folder splits are ",self.image_fol.split('/'))
        # print("Current fol number is ", current_fol_number)
        if past_len<counter_fol_past_len:
            traj = counter_fol_past_traj
        length_unique, traj = get_traj_length_unique_actual(traj)
        past_len_now, past_traj = get_traj_length_unique_actual(past_traj)
        past_len_now = past_traj.shape[0]
        if traj.shape[0]-past_len_now<fixed_len:
            if past_len_now<traj.shape[0]:
                traj = traj[past_len_now-1:,:]
            else:
                traj = traj[-1:,:]
            # traj = traj.astype(int)
            
            length_unique, traj = get_traj_length_unique_actual(traj)
            traj = traj_interp(traj)
            pad_len = fixed_len - traj.shape[0]    
            pad_list = []
            for i in range(int(np.ceil(pad_len))):
                if (i < pad_len):
                    pad_list.append([traj[-1,0], traj[-1,1]])
                else:
                    pad_list.append([np.NaN, np.NaN])
            pad_array = np.array(pad_list[:pad_len])
            if pad_len>0:
                output = np.vstack((traj, pad_array))
            else:
                output = traj
            return  output.astype(float)
        traj = traj[past_len_now:past_len_now+fixed_len,:]
        # traj = traj.astype(int)
        traj = traj_interp(traj)
        length_unique, traj = get_traj_length_unique_actual(traj)
        traj = traj_interp(traj)
        pad_len = fixed_len - traj.shape[0]    
        pad_list = []
        for i in range(int(np.ceil(pad_len))):
            if (i < pad_len):
                pad_list.append([traj[-1,0], traj[-1,1]])
            else:
                pad_list.append([np.NaN, np.NaN])
        pad_array = np.array(pad_list[:pad_len])
        if pad_len>0:
            output = np.vstack((traj, pad_array))
        else:
            output = traj
        return  output.astype(float)
        if not is_valid_traj(traj):
            print("Invalid traj ", traj, current_fol_number, counter)
            # embed()
        return traj

    def auto_pad_future(self, traj, counter):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = FIXED_LEN
        current_fol_number = int(self.image_fol.split('/')[-1])
        # print("Current fol number is ", current_fol_number)
        # if traj.shape[0] >= fixed_len:
            # if current_fol_number+fixed_len>counter and current_fol_number<counter:
            #     embed()
            #     traj = traj[current_fol_number:counter,:]
            #     pad_len = fixed_len - traj.shape[0]    
            #     pad_list = []
            #     for i in range(int(np.ceil(pad_len))):
            #         if (i < pad_len):
            #             # pad_list.append([(traj[-1,0]-1), traj[-1,1]])
            #             pad_list.append([traj[-1,0], traj[-1,1]])
            #         else:
            #             pad_list.append([np.NaN, np.NaN])
            #     # print(pad_list)
            #     pad_array = np.array(pad_list[:pad_len])
            #     if pad_len>0:
            #         output = np.vstack((traj, pad_array))
            #     else:
            #         output = traj
            #     return output

        if current_fol_number+fixed_len <=traj.shape[0]:
            if current_fol_number<counter and (traj[current_fol_number,:] == traj[counter,:]).all():
                traj = np.ones((fixed_len, 2))*traj[current_fol_number,:].T
                traj = traj.astype(int)
                return traj
            traj = traj[current_fol_number:current_fol_number+fixed_len,:]
            return traj
        else:
            # traj = np.ones((fixed_len, 2))*traj[current_fol_number,:].T
            # traj = traj.astype(int)
            # return traj
            traj = traj[current_fol_number:traj.shape[0], :]
            traj = traj.astype(int)
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
       
            pad_len = fixed_len - traj.shape[0]    
            pad_list = []
            for i in range(int(np.ceil(pad_len))):
                if (i < pad_len):
                    # pad_list.append([(traj[-1,0]-1), traj[-1,1]])
                    pad_list.append([traj[-1,0], traj[-1,1]])
                else:
                    pad_list.append([np.NaN, np.NaN])
            pad_array = np.array(pad_list[:pad_len])
            if pad_len>0:
                output = np.vstack((traj, pad_array))
            else:
                output = traj
            return  output