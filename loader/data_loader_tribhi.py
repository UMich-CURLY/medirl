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

np.set_printoptions(threshold=np.inf, suppress=True)
import random
import copy
import math


class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/root/medirl/data/irl_data_dec_11_rl_agent', pre_train=False, tangent=False,
                 more_kinematic=None):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        # if train:
        #     self.data_dir = datadir + 'train_data/'
        # else:
        #     self.data_dir = datadir + 'test_data/'

        # if demo is not None:
        #     self.data_dir = datadir + '/irl_data/' + demo
        self.data_dir = datadir
        items = os.listdir(self.data_dir)
        self.data_list = []
        for item in items:
            image_fol = self.data_dir + '/' + item+"/traj_fixed.npy"
            if not (os.path.exists(image_fol)):
                continue
            if (self.check_isnone(self.data_dir + '/' + item)):
                continue
            self.data_list.append(self.data_dir + '/' + item)


        #self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train

        # kinematic related feature
        self.center_idx = self.grid_size / 2
    
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
        with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
            traj = np.load(f)

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
                return None, None
            # print("for i mean is std is ", i, np.mean(feat[i]), np.std(feat[i]))
            # print("After normalize min max", np.min(feat[i]), np.max(feat[i]))       
 
    
        traj = self.auto_pad_future(traj[:, :2])
        return feat, traj

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