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

FIXED_LEN = 20
def Dataloader_by_Index(data_loader, target=0):
    for index, data in enumerate(data_loader):
        if index == target:
            return data
    return None

def visualize(input):
    return None
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
    return d
def get_traj_length_unique(traj):
    lengths = []
    traj_list = []
    for j in range(len(traj)):
        # if list(traj[j]) not in traj_list:
        if True:
            traj_list.append([traj[j][0], traj[j][1]])
    lengths.append(len(traj_list))
    return np.array(lengths), np.array(traj_list)
def auto_pad_future_from_past_counter(fol_path, traj, past_traj, counter ):
    fixed_len = FIXED_LEN
    past_len = past_traj.shape[0]
    print("Past len is ", past_len)
    counter_fol = fol_path + '/' + str(counter)
    counter_fol_past_traj = np.load(counter_fol+"/robot_past_traj.npy")
    counter_fol_past_traj = traj_interp(counter_fol_past_traj)
    lengh, counter_fol_past_traj = get_traj_length_unique(counter_fol_past_traj)
    counter_fol_past_len = counter_fol_past_traj.shape[0]
    print("Counter past len is ", counter_fol_past_len)
    
    if past_len<counter_fol_past_len:
        traj = counter_fol_past_traj
    # if past_traj[-1][0] == counter_fol_past_traj[-1][0] and past_traj[-1][1] == counter_fol_past_traj[-1][1]:
    #     traj = counter_fol_past_traj
    if traj.shape[0]-past_len<fixed_len:
        if past_len<traj.shape[0]:
            traj = traj[past_len-1:,:]
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
        # embed()
    return traj

exp_name = '6.33'
grid_size = 60
discount = 0.9
lr = 5e-4
n_epoch = 16
batch_size = 8
n_worker = 2
use_gpu = True
assert grid_size % 2 == 0, "grid size must be even number"
data_dir = "data/irl_jul_19_5/train"
demos =  os.listdir(data_dir)
demos.remove('metrics_data.csv')
demos.sort(key=lambda x:int(x[5:]))
data_list = []
remove_list = ['traj.npy', 'rank.txt', 'final_overlayed_map.png', 'crossing_count.txt', 'ep_count.txt', 'new_rank.txt', 'episode_1.gif']
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8090)
env='v{}-{}'.format(exp_name+"robot", host)
for demo in demos:
    items = os.listdir(data_dir+"/"+demo)
    # if (int(demo[5:])<27):
    #     continue
    rank_path = data_dir+"/"+demo + '/rank.txt'
    new_rank_path = data_dir+"/"+demo + '/new_rank.txt'
    with open(rank_path, 'r') as f:
        rank = f.read()
        f.close()
        rank = np.round(float(rank), 3)
    print(rank)
    metric = {}
    metrics = []
    with open(data_dir+"/metrics_data.csv", mode='r', newline='') as file:
            # Create a DictReader object
            csv_reader = csv.DictReader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Print each row as a dictionary
                metrics.append(row)
    metric = metrics[int(demo[5:])]
    print("Metric is ", metric)
    items = [item for item in items if item.isdigit()]
    items.sort(key= lambda x:int(x))
    for item in items:
        if item in remove_list:
            continue
        img_path = data_dir+"/"+demo +'/' + item+ '/grid_map.png'
        sized = (256,256)
        img_grid = Image.open(img_path)
        
        with open(data_dir+'/'+demo+'/'+item+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        # if len(full_traj) == 0:
        #     with open(self.image_fol+"/traj_fixed.npy", 'rb') as f:
        #         full_traj = np.load(f)
        full_traj = traj_interp(full_traj)
        length, robot_past_traj = get_traj_length_unique(full_traj)
        with open(data_dir+'/'+demo+"/traj.npy", 'rb') as f:
            full_traj = np.load(f)
        full_traj = np.array(traj_interp(full_traj), np.int)
        # print("Valid full traj? ", is_valid_traj(full_traj))
        length, robot_traj = get_traj_length_unique(full_traj)
        file = open(data_dir+'/'+demo+ '/new_crossing_count.txt', 'r')
        counter_crossing = int(file.read())
        robot_traj = auto_pad_future_from_past_counter(data_dir+'/'+demo, robot_traj[:, :2], robot_past_traj, counter_crossing)
        # img_tensor = torch.from_numpy(img)
        for xy in robot_traj:
            img_grid.putpixel((int(xy[0]), int(xy[1])), (255, 0, 0))
        if metric["social_nav_to_pos_success"] == 'True' or metric["social_nav_to_pos_success"] == 'TRUE':
            a = 1
        else:
            a = 0
        print("Metric is " , int(a))
        img_grid = img_grid.resize(sized)
        img = np.array(img_grid)
        if abs(int(item) - counter_crossing) < 20:
            img_callback_win = vis.image(img.T, opts= dict(caption='count {}, cc {}, succ {} '.format(item,counter_crossing, metric["social_nav_to_pos_success"])))
        new_rank = a*rank
        with open(new_rank_path, 'w') as f:
            f.write(str(new_rank))
            f.close()
    # new_crossing_count = input ('Enter crossing count for demo {} '.format(demo[5:]))
    # with open(data_dir+'/'+demo+ '/new_crossing_count.txt', 'w') as f:
    #     f.write(new_crossing_count)
    #     f.close()    
    # new_rank = input ('Enter rank for demo {} '.format(demo[5:]))
    # if new_rank == '':
    #     continue
    # with open(rank_path, 'w') as f:
    #     f.write(new_rank)
    #     f.close()
    
    #     def img_click_callback(event):
    #         img_coord_text = vis.text("Coords: ", env = env)
    #         if event['event_type'] != 'Click':
    #             return

    #         coords = "x: {}, y: {};".format(
    #             event['image_coord']['x'], event['image_coord']['y']
    #         )
    #         img_coord_text = vis.text(coords, win=img_coord_text, append=True, env = env )

    #     vis.register_event_handler(img_click_callback, img_callback_win)
        # embed()
# for index, (feat, robot_traj, human_past_traj, robot_past_traj, demo_rank) in enumerate(train_loader_robot):
#         print("outside loop")
#         start = time.time()
