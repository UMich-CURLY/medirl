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


class OffroadLoader(Dataset):
    def __init__(self, grid_size, train=True, demo=None, datadir='/root/medirl/data/', pre_train=False, tangent=False,
                 more_kinematic=None):
        assert grid_size % 2 == 0, "grid size must be even number"
        self.grid_size = grid_size
        self.image_fol = datadir + "demo_0"
        #self.data_normalization = sio.loadmat(datadir + '/irl_data/train-data-mean-std.mat')
        self.pre_train = pre_train

        # kinematic related feature
        self.center_idx = self.grid_size / 2
    

    def __getitem__(self, index):
        goal_sink_feat = np.array(Image.open(self.image_fol+"/goal_sink.png")).T
        goal_sink_feat = goal_sink_feat/255*np.ones(goal_sink_feat.shape)
        temp_sink_feat = np.zeros([1,goal_sink_feat.shape[1], goal_sink_feat.shape[2]])
        for i in range(goal_sink_feat.shape[1]):
            for j in range(goal_sink_feat.shape[2]):
                if (goal_sink_feat[1,i,j] == 1):
                    temp_sink_feat[0,i,j] = -1
                if (goal_sink_feat[2,i,j] == 1):
                    temp_sink_feat[0,i,j] = 1
        goal_sink_feat = temp_sink_feat
        semantic_img_feat = np.array(Image.open(self.image_fol+"/semantic_img.png"))[:,:,0:3].T
        with open(self.image_fol+"/trajectory.npy", 'rb') as f:
            traj = np.load(f)

        for i in range(traj.shape[0]):
            temp = traj[i,0] 
            traj[i,0]= traj[i,1]
            traj[i,1] = temp 
        # visualize rgb
        feat = np.concatenate((goal_sink_feat, semantic_img_feat), axis = 0)
        # normalize features locally

        for i in range(feat.shape[0]):
            print(feat[i].shape)
            print("Before nomalize", np.max(feat[i]))  
            feat[i] = feat[i] - np.mean(feat[i])*np.ones(feat[i].shape)
            feat[i] = feat[i] / np.std(feat[i])*np.ones(feat[i].shape)
            print("for i mean is std is ", i, np.mean(feat[i]), np.std(feat[i]))
            print("After normalize min max", np.min(feat[i]), np.max(feat[i]))       
 
    

        return feat, traj

    def __len__(self):
        return 1

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