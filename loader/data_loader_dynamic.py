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
    
def is_valid_traj(traj):
    i = 0
    while i < len(traj)-1:
        if not check_neighbor(traj[i], traj[i+1]):
            return False
        i = i+1
    return True

class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/root/medirl/data/irl_feb_6', pre_train=False, tangent=False,
                 more_kinematic=None, human = False):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        # if train:
        #     self.data_dir = datadir + '/train'
        # else:
        #     self.data_dir = datadir + '/test'
        # print(self.data_dir)
        # if not train:
        #     self.data_dir = '/root/medirl/data/irl_driving_full_future'
        # if demo is not None:
        #     self.data_dir = datadir + '/irl_data/' + demo
        items = os.listdir(self.data_dir)
        self.data_list = []
        for item in items:
            past_traj = self.data_dir + '/' + item+"/past_traj.npy"
            if not (os.path.exists(past_traj)):
                continue
            if (self.check_isnone(self.data_dir + '/' + item)):
                continue
            future_traj = self.data_dir + '/' + item+"/future_traj.npy"
            if not (os.path.exists(future_traj)):
                continue
            if (self.check_isnone(self.data_dir + '/' + item)):
                continue
            self.data_list.append(self.data_dir + '/' + item)


        #self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train

        # kinematic related feature
        self.center_idx = self.grid_size / 2
        self.is_human = human
    
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
    
    



    def __getitem__(self, index):
        self.image_fol = self.data_list[index]
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
        with open(self.image_fol+"/past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
            past_full_traj = full_traj
        past_other_traj = []
        if self.is_human:
            past_traj = full_traj[:,:,1]
            past_other_traj = full_traj[:,:,0]
        else:
            past_traj = full_traj[:,:,0]
            past_other_traj = full_traj[:,:,1]
        past_traj = transpose_traj(past_traj)
        past_other_traj = transpose_traj(past_other_traj)
        if not is_valid_traj(past_traj) or not is_valid_traj(past_other_traj):
            print("Bad past traj in ", self.image_fol)
            # embed()
        with open(self.image_fol+"/future_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        # if len(full_traj) == 0:
        #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
        #         full_traj = np.load(f)
        future_other_traj = []
        if self.is_human:
            future_traj = full_traj[:,:,1]
            future_other_traj = full_traj[:,:,0]
        else:
            future_traj = full_traj[:,:,0]
            future_other_traj = full_traj[:,:,1]
        future_traj = transpose_traj(future_traj)
        future_other_traj = transpose_traj(future_other_traj)
        if not is_valid_traj(future_traj) or not is_valid_traj(future_other_traj):
            print("Bad future traj in ", self.image_fol)
            # embed()
        
        # visualize rgb
        # with open(self.image_fol+"/sdf.npy", 'rb') as f:
        #     sdf_feat = np.load(f)
        feat = np.concatenate((goal_sink_feat, semantic_img_feat), axis = 0)
        ### Add the traj features 
        self_traj_feature = np.zeros(goal_sink_feat.shape)
        self_traj_feature[0,list(np.array(past_traj[:,0], np.int)), list(np.array(past_traj[:,1], np.int))] = 100
        other_traj_feature = np.zeros(goal_sink_feat.shape)
        
        other_traj_feature[0,list(np.array(past_other_traj[:,0], np.int)), list(np.array(past_other_traj[:,1], np.int))] = 100
        kin_feats = np.concatenate((self_traj_feature, other_traj_feature), axis = 0)
        # feat = np.concatenate((feat, kin_feats), axis = 0)
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
 
    
        past_traj = self.auto_pad_past(past_traj[:, :2])
        future_traj = self.auto_pad_future(future_traj[:, :2])
        past_other_traj = self.auto_pad_past(past_other_traj[:, :2])
        future_other_traj = self.auto_pad_future(future_other_traj[:, :2])
        return feat, past_traj, future_traj, past_other_traj, future_other_traj

    def __len__(self):
        return len(self.data_list)

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
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output

    def auto_pad_future(self, traj):
        """
        add padding (NAN) to traj to keep traj length fixed.
        traj shape needs to be fixed in order to use batch sampling
        :param traj: numpy array. (traj_len, 2)
        :return:
        """
        fixed_len = self.grid_size
        if traj.shape[0] >= self.grid_size:
            traj = traj[:self.grid_size, :]
            #raise ValueError('traj length {} must be less than grid_size {}'.format(traj.shape[0], self.grid_size))
        pad_len = self.grid_size - traj.shape[0]
        pad_array = np.full((pad_len, 2), np.nan)
        output = np.vstack((traj, pad_array))
        return output