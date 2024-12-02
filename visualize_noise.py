from PIL import Image
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
    raw_img = np.array(Image.open(image_fol+"/raw_img.png"))[:,:,0:3]
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

def annotate_raw_image(raw_img, robot_traj_grid):
    grid_img_res = 0.1
    img_res = 6/raw_img.shape[0]
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
    i =0 
    for point in new_points:
        raw_img[point[0], point[1]] = [0,0,color[i]]
        i += 1
    # grid_img[point[0], point[1]] = [255,0,0]
    raw_img[point[0], point[1]] = [0, 255, 0]
    return raw_img

def main():
    data_folder = "data/irl_sept_24_3_new_cross_noised/train/demo_3"
    for sub_folder in os.listdir(data_folder):
        if not os.path.isdir(data_folder + "/" + sub_folder):
            continue
        folder = data_folder + "/" + sub_folder
        grid_img = get_grid_map(folder)
        raw_img = get_raw_img(folder)
        for noise_counter in range(0, 6):
            robot_traj_grid = get_robot_traj(folder, noise_counter)
            grid_img = annotate_image(grid_img, robot_traj_grid)
            raw_img = annotate_raw_image(raw_img, robot_traj_grid)
        Image.fromarray(np.uint8(grid_img)).save(folder+"/full_noise_overlay.png")
        Image.fromarray(np.uint8(raw_img)).save(folder+"/full_noise_overlay_raw.png")
        
        try: 
            robot_traj_no_noise = np.load(folder+"/robot_"+"traj_post.npy")
            robot_traj_no_noise = transpose_traj(robot_traj_no_noise[0])
            grid_img = get_grid_map(folder)
            grid_img = annotate_image(grid_img, np.array(robot_traj_no_noise, dtype = int))
            Image.fromarray(np.uint8(grid_img)).save(folder+"/full_noise_overlay_no_noise.png")
            print(folder)
        except:
            pass
if __name__ == "__main__":
    main()

