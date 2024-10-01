import mdp.offroad_grid as offroad_grid
from loader.data_loader_rank import OffroadLoader
from torch.utils.data import DataLoader
import numpy as np

np.set_printoptions(threshold=np.inf)  # print the full numpy array
import visdom
import warnings
import logging
import os
from PIL import Image
import torch
from torch.autograd import Variable
import time
from maxent_irl_social import pred, rl, overlay_traj_to_map, visualize, visualize_batch
from IPython import embed
import csv
logging.basicConfig(filename='maxent_irl_social.log', format='%(levelname)s. %(asctime)s. %(message)s',
                    level=logging.DEBUG)
def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader):
        if index == target:
            return data
    return None

def visualize(input):
    return None

exp_name = '6.33'
grid_size = 60
discount = 0.9
lr = 5e-4
n_epoch = 16
batch_size = 8
n_worker = 2
use_gpu = True
assert grid_size % 2 == 0, "grid size must be even number"
data_dir = "data/single_ep/train"
demos =  os.listdir(data_dir)
demos.remove('metrics_data.csv')
demos.sort(key=lambda x:int(x[5:]))
data_list = []
remove_list = ['traj.npy', 'rank.txt', 'final_overlayed_map.png']
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8098)
env='v{}-{}'.format(exp_name+"robot", host)
num_item = {}
min_num_items = 1000
for demo in demos:
    items = os.listdir(data_dir+"/"+demo)
    items = list([ int(x) for x in items if x.isdigit() ])
    print("Items are ", np.max(items))
    num_item.update({demo: np.max(items)})
    if np.max(items)<min_num_items:
        min_num_items = np.max(items)

for demo in num_item.keys():
    items_to_remove = num_item[demo] - min_num_items
    print("Items to remove are ", items_to_remove)
    to_keep = np.random.choice(num_item[demo], min_num_items, replace=False)
    file = open(data_dir+'/'+ demo +'/new_crossing_count.txt', 'r')
    new_counter_crossing = int(file.read())
    file.close()
    if new_counter_crossing not in to_keep:
        to_keep = np.append(to_keep, new_counter_crossing)
    to_keep = np.sort(to_keep)
    print("To keep are ", to_keep)

    with open(data_dir+'/'+demo+"/tokeep.npy", 'wb') as f:
        np.save(f, np.array(to_keep))
