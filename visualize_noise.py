from PIL import Image, ImageDraw
import numpy as np
from IPython import embed
import os
def get_grid_map(image_fol):
    grid_img = np.array(Image.open(image_fol+"/grid_map.png"))[:,:,0:3]
    return grid_img

def get_robot_traj(image_fol, noise_num):
    if noise_num == 0:
        robot_traj = np.load(image_fol+"/robot_noise_"+"traj.npy")
    else:
        robot_traj = np.load(image_fol+"/robot_noise_"+"traj"+str(noise_num-1)+".npy")
    robot_traj = traj_interp(robot_traj)
    return robot_traj

def get_raw_img(image_fol):
    raw_img = np.array(Image.open(image_fol+"/new_obs_raw.png"))[:,:,0:3]
    return raw_img


def grid_coord_to_raw(coord, img_res, grid_img_res):
    factor = grid_img_res/img_res
    # factor = 1/factor
    return [int((60-coord[0])*factor), int((60-coord[1])*factor)]

def raw_to_grid(coord, img_res, grid_img_res):
    factor = grid_img_res/img_res
    # factor = 1/factor
    return [60-int(coord[0]/factor), 60-int(coord[1]/factor)]


def transpose_traj(traj):
    for i in range(traj.shape[0]):
        temp = traj[i,0] 
        traj[i,0]= traj[i,1]
        traj[i,1] = temp 
    return traj


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
   
    return np.array(d)

def annotate_image(grid_img, robot_traj_grid):
    num_of_points = len(robot_traj_grid)
    color = np.linspace(0, 255, num_of_points)
    i =0 
    for point in robot_traj_grid:
        if point[0] <0:
            point[0] = 0
        if point[1] <0:
            point[1] = 0
        if point[0] >= 60:
            point[0] = 59
        if point[1] >= 60:
            point[1] = 59
        grid_img[point[0], point[1]] = [255-color[i],0,color[i]]
        i += 1
    # grid_img[point[0], point[1]] = [255,0,0]
    return grid_img

def annotate_raw_image(raw_img, robot_traj_grid, raw_img_size, counter):
    grid_img_res = 0.1
    img_res = 6/raw_img_size
    num_of_points = len(robot_traj_grid)
    
    new_points = []
    for point in robot_traj_grid:
        if point[0] <0:
            point[0] = 0
        if point[1] <0:
            point[1] = 0
        if point[0] >= 60:                                                                          
            point[0] = 59
        if point[1] >= 60:
            point[1] = 59
        point = grid_coord_to_raw(point, img_res, grid_img_res)
        new_points.append(point)
    new_points = traj_interp(np.array(new_points))
    color = np.linspace(0, 255, new_points.shape[0])
    # i =0 
    # for point in new_points:
    #     raw_img[point[0], point[1]] = [0,0,color[i]]
    #     i += 1
    # # grid_img[point[0], point[1]] = [255,0,0]
    # raw_img[point[0], point[1]] = [0, 255, 0]
    colors = [
        (255, 255, 0),    # Bright Yellow
        (0, 255, 255),    # Cyan
        (255, 0, 255),    # Magenta
        (0, 255, 0),      # Lime Green
        (255, 165, 0),    # Orange
        (255, 255, 255),   # White
        (255, 0 , 0)
    ]
    marker_color = (255, 0, 0)  # Red color
    marker_size = 0  # Half the length of the "X" arms
    line_width = 3  # Line thickness
    raw_img = Image.fromarray(np.uint8(raw_img))
    draw = ImageDraw.Draw(raw_img)
    for i in range(len(new_points)-1):
        [y1,x1] = new_points[i]
        [y2,x2] = new_points[i+1]
        draw.line((x1 - marker_size, y1 - marker_size, x2 + marker_size, y2 + marker_size), fill=colors[counter], width=line_width)
        
    return raw_img

def main():
    data_folder = "data/irl_sept_24_3_new_cross_noised/train/demo_3"
    for sub_folder in os.listdir(data_folder):
        if not os.path.isdir(data_folder + "/" + sub_folder):
            continue
        folder = data_folder + "/" + sub_folder
        grid_img = get_grid_map(folder)
        raw_img = get_raw_img(folder)
        raw_img_size = raw_img.shape[0]
        for noise_counter in range(0, 6):
            robot_traj_grid = get_robot_traj(folder, noise_counter)
            grid_img = annotate_image(grid_img, robot_traj_grid)
            raw_img = annotate_raw_image(raw_img, robot_traj_grid, raw_img_size, noise_counter)
        Image.fromarray(np.uint8(grid_img)).save(folder+"/full_noise_overlay.png")
        raw_img.save(folder+"/full_noise_overlay_raw.png")
        
        with open(folder+"/robot_past_traj.npy", 'rb') as f:
            full_traj = np.load(f)
        
        robot_past_traj = full_traj
        file = open(folder+ '/new_crossing_count.txt', 'r')
        counter_crossing_data = file.read().split('\n')
        number_of_stops = len(counter_crossing_data)
        current_fol_number = int(folder.split('/')[-1])
        # print("Number of stops ", number_of_stops, counter_crossing_data)
        for counter_crossing in counter_crossing_data:
            counter_crossing = int(counter_crossing)
            if counter_crossing >= current_fol_number:
                break
        counter_fol = data_folder + '/' + str(counter_crossing)
        print("counter_fol", counter_fol)
        robot_past_at_crossing = np.load(counter_fol+"/robot_past_traj.npy")
        robot_future = robot_past_at_crossing[len(robot_past_traj):]
        robot_traj_no_noise = robot_future
        robot_traj_no_noise = transpose_traj(robot_traj_no_noise)
        grid_img = get_grid_map(folder)
        grid_img = annotate_image(grid_img, np.array(robot_traj_no_noise, dtype = int))
        Image.fromarray(np.uint8(grid_img)).save(folder+"/full_noise_overlay_no_noise.png")
        raw_img = get_raw_img(folder)
        raw_img = annotate_raw_image(raw_img, robot_traj_no_noise, raw_img_size, 6)
        raw_img.save(folder+"/overlay_raw_no_noise.png")
        print(folder)

if __name__ == "__main__":
    main()

