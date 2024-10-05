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
data_dir = "data/irl_sept_24_3/test"
demos =  os.listdir(data_dir)
demos.remove('metrics_data.csv')
demos.sort(key=lambda x:int(x[5:]))
data_list = []
remove_list = ['traj.npy', 'rank.txt', 'final_overlayed_map.png']
host = os.environ['HOSTNAME']
vis = visdom.Visdom(env='v{}-{}'.format(exp_name+"robot", host), server='http://127.0.0.1', port=8098)
env='v{}-{}'.format(exp_name+"robot", host)
for demo in demos:
    items = os.listdir(data_dir+"/"+demo)
    items = [ x for x in items if x.isdigit() ]
    items.sort(key=lambda x:int(x))
    new_rank_path = data_dir+"/"+demo + '/new_rank.txt'
    metric = {}
    ep_list = {}
    metrics = []
    with open(data_dir+"/metrics_data.csv", mode='r', newline='') as file:
            # Create a DictReader object
            csv_reader = csv.DictReader(file)

            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Print each row as a dictionary
                metrics.append(row)
    for metric in metrics:
        if metric["social_nav_to_pos_success"] == 'True' or metric["social_nav_to_pos_success"] == 'TRUE':
            if metric['ep_no'] in ep_list.keys():
                # try:
                new_list = ep_list[metric['ep_no']].copy()
                new_list.append(int(metric['actual_num_steps']))
                # self.ep_list.update({metric['ep_no']: [self.ep_list[metric['ep_no']][0:-1][0], (int(metric['reset_counter']))]})
                ep_list.update({metric['ep_no']: new_list})
                # except:
                
            else:
                ep_list.update({metric['ep_no']: [int(metric['actual_num_steps'])]})
    
    min_steps_dict = {}
    for ep in ep_list:
        min_steps_dict[ep] = np.min(ep_list[ep])
    metric = metrics[int(demo[5:])]
    print("Metric is ", metric)
    vels = []
    # if not (metric["social_nav_to_pos_success"] == 'True' or metric["social_nav_to_pos_success"] == 'TRUE'):
    #     continue
    for item in items:
        if item in remove_list:
            continue
        if item == '0':
            img_path = data_dir+"/"+demo + "/" +item + '/grid_map.png'
            sized = (256,256)
            img_grid = Image.open(img_path)
            img_grid = img_grid.resize(sized)
            img = np.array(img_grid)
            img_callback_win = vis.image(img.T, opts= dict(caption='count {}, item {}, succ {} '.format(item,demo, metric["social_nav_to_pos_success"])))
        if item == str(max([int(x) for x in items])-1):
            img_path = data_dir+"/"+demo + "/" +item + '/grid_map.png'
            sized = (256,256)
            img_grid = Image.open(img_path)
            img_grid = img_grid.resize(sized)
            img = np.array(img_grid)
            img_callback_win = vis.image(img.T, opts= dict(caption='count {}, item {}, succ {} '.format(item,demo, metric["social_nav_to_pos_success"])))
        img_path = data_dir+"/"+demo + "/" +item + '/new_overlayed_grid_map.png'
        sized = (256,256)
        img_grid = Image.open(img_path)
        img_grid = img_grid.resize(sized)
        img = np.array(img_grid)
        with open(data_dir+"/"+demo +"/"+ item + "/vels.npy", 'rb') as f:
            vels_read = np.load(f)
        vels.append(vels_read)
        # img_tensor = torch.from_numpy(img)
        if metric["social_nav_to_pos_success"] == 'True' or metric["social_nav_to_pos_success"] == 'TRUE':
            a = 1
            num_steps = metric['actual_num_steps']
            ep_number = metric['ep_no']
            minimum_num_steps = min_steps_dict[ep_number]
            rank = a * int(minimum_num_steps)/int(num_steps)
        else:
            a = 0
            rank = a
        # img_callback_win = vis.image(img.T, opts= dict(caption='demo {}, item {}, succ {} '.format(metric["ep_no"],item, metric["social_nav_to_pos_success"])))
        new_rank = rank
        with open(new_rank_path, 'w') as f:
            f.write(str(new_rank))
            f.close()
    img_callback_win = vis.image(img.T, opts= dict(caption='count {}, item {}, succ {} '.format(item,demo, metric["social_nav_to_pos_success"])))
            

    started = False
    counter_stop = 0
    potential_crossing = 0
    counter_cross = []
    same_stop = False
    counter = 0
    for vel in vels:
        robot_vel = vel[0]
        if started is False and robot_vel != 0.0:
            started = True
        else:
            if started is False:
                counter += 1
                continue
        # print("Started at ", counter)
        if robot_vel == 0.0 and same_stop is False:
            counter_stop = 0 
            potential_crossing = counter 
            same_stop = True
            # print("Looking into potential crossing at ", counter)
        if robot_vel == 0.0 and same_stop is True:
            counter_stop += 1
            # print("Counter stop is ", counter_stop)
        if robot_vel != 0.0 and same_stop is True:
            same_stop = False
            # print("Writing potential crossing at ", potential_crossing)
            if counter_stop > 15:
                counter_cross.append(counter-1)
        counter += 1
    counter_cross.append(counter-2)
    print("Counter cross is ", counter_cross)
    
    new_list = [str(c) for c in counter_cross]
    with open(data_dir+"/"+demo + "/" +'new_crossing_count.txt', mode='wt', encoding='utf-8') as myfile:
        myfile.write('\n'.join(new_list))

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
