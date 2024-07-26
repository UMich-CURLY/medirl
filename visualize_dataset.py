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
data_dir = "data/raw_data/irl_jun_26_2"
demos =  os.listdir(data_dir)
demos.sort()
data_list = []
remove_list = ['traj.npy', 'rank.txt', 'final_overlayed_map.png']
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8098)
env='v{}-{}'.format(exp_name+"robot", host)
for demo in demos:
    items = os.listdir(data_dir+"/"+demo)
    rank_path = data_dir+"/"+demo + '/rank.txt'
    try:
        with open(rank_path, 'r') as f:
            rank = int(f.read(1))
            f.close()
    except:
        rank = '5'
    items.sort()
    for item in items:
        if item in remove_list:
            continue
        img_path = data_dir+"/"+demo + '/0/grid_map.png'
        sized = (256,256)
        img_grid = Image.open(img_path)
        img_grid = img_grid.resize(sized)
        img = np.array(img_grid)
        # img_tensor = torch.from_numpy(img)
        img_callback_win = vis.image(img.T, opts= dict(caption='demo {}, item {}, rank {},  overlayed_img'.format(demo[5:], item, rank)))
        break
    new_rank = input ('Enter rank for demo {} '.format(demo[5:]))
    if new_rank == '':
        continue
    with open(rank_path, 'w') as f:
        f.write(new_rank)
        f.close()
    
        def img_click_callback(event):
            img_coord_text = vis.text("Coords: ", env = env)
            if event['event_type'] != 'Click':
                return

            coords = "x: {}, y: {};".format(
                event['image_coord']['x'], event['image_coord']['y']
            )
            img_coord_text = vis.text(coords, win=img_coord_text, append=True, env = env )

        vis.register_event_handler(img_click_callback, img_callback_win)
        # embed()
# for index, (feat, robot_traj, human_past_traj, robot_past_traj, demo_rank) in enumerate(train_loader_robot):
#         print("outside loop")
#         start = time.time()
