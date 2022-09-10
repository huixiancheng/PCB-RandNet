import time
from matplotlib import pyplot as plt
from utils.data_process import DataProcessing as DP
from config import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import pickle
from tqdm import tqdm

name = 'SemanticKITTI'
dataset_path = '/home/chx/Work/semantic-kitti-randla/sequences'

# name = 'SemanticPOSS'
# dataset_path = '/home/chx/Work/SemanticPOSS_dataset_grid_0.01/sequences'

num_classes = cfg.num_classes
ignored_labels = np.sort([0])

mode = 'training'
sampling = ['random', 'polar', 'fps']
# sampling_way = 'random'


seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
data_list = DP.get_file_list(dataset_path, seq_list)
data_list = sorted(data_list)
print("Using {} scans from sequences {}".format(len(data_list), seq_list))


import random
seed = 2
random.seed(seed)
random.shuffle(data_list)


def get_data(file_path):
    seq_id = file_path[0]
    frame_id = file_path[1]
    kd_tree_path = join(dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
    # read pkl with search tree
    with open(kd_tree_path, 'rb') as f:
        search_tree = pickle.load(f)
    points = np.array(search_tree.data, copy=False)
    # load labels
    label_path = join(dataset_path, seq_id, 'labels', frame_id + '.npy')
    labels = np.squeeze(np.load(label_path))
    return points, search_tree, labels
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)
def polar_samplr(mid_xyz, grid):
    xyz_pol = cart2polar(mid_xyz)

    max_bound_r = min(np.percentile(xyz_pol[:, 0], 100, axis=0), 50)
    min_bound_r = max(np.percentile(xyz_pol[:, 0], 0, axis=0), 3)
    max_bound_p = np.max(xyz_pol[:, 1], axis=0)
    min_bound_p = np.min(xyz_pol[:, 1], axis=0)
    max_bound_z = min(np.max(xyz_pol[:, 2], axis=0), 1.5)
    min_bound_z = max(np.min(xyz_pol[:, 2], axis=0), -3)
    max_bound = np.concatenate(([max_bound_r], [max_bound_p], [max_bound_z]))
    min_bound = np.concatenate(([min_bound_r], [min_bound_p], [min_bound_z]))
    cur_grid_size = np.asarray(grid)
    crop_range = max_bound - min_bound
    intervals = crop_range / (cur_grid_size - 1)

    if (intervals == 0).any(): print("Zero interval!")

    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
    keys, revers, counts = np.unique(grid_ind, return_inverse=True, return_counts=True, axis=0)

    idx = np.argsort(revers)
    mid_xyz = mid_xyz[idx]
    slic = counts.cumsum()
    slic = np.insert(slic, 0, 0)
    left_xyz, right_xyz = np.zeros([0, 3]), np.zeros([0, 3])

    small = counts[counts < 4]
    new_nums = len(mid_xyz) // 4 - sum(small)
    new_grid = len(counts) - len(small)
    sample_list = []
    for i in range(new_grid):
        curr = new_nums // new_grid
        sample_list.append(curr)
        new_nums -= curr
        new_grid -= 1
    sample_list = np.array(sample_list)
    sample_list = DP.shuffle_idx(sample_list)
    idx = 0
    for i in range(len(counts)):
        select_xyz = mid_xyz[slic[i]:slic[i + 1]]
        np.random.shuffle(select_xyz)
        nubs = counts[i]
        if nubs >= 4:
            downs_n = sample_list[idx]
            idx += 1
            # downs_n = new_num_points_each_grid
            left_xyz = np.concatenate((left_xyz, select_xyz[0:downs_n]), axis=0)
            right_xyz = np.concatenate((right_xyz, select_xyz[downs_n:]), axis=0)
        else:
            left_xyz = np.concatenate((left_xyz, select_xyz), axis=0)

    supp = len(mid_xyz) // 4 - len(left_xyz)
    if supp == 0:
        pass
    elif supp > 0:
        np.random.shuffle(right_xyz)
        left_xyz = np.concatenate((left_xyz, right_xyz[0:supp]))
        right_xyz = right_xyz[supp:]
    else:
        np.random.shuffle(right_xyz)
        supp = len(mid_xyz) // 4
        left_xyz, supp_xyz = left_xyz[:supp], left_xyz[supp:]
        right_xyz = np.concatenate((supp_xyz, right_xyz))

    np.random.shuffle(left_xyz)
    return left_xyz
    # np.random.shuffle(right_xyz)
    # select_points = np.concatenate((left_xyz, right_xyz), axis=0)
    # return select_points
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

# for sampling_way in sampling:


a = [11264, 2816, 704, 176]

start = 2
for start in range(1, 5):
    for sampling_way in sampling:
        test = []
        for list in range(30):
            points, search_tree, labels = get_data(data_list[list])
            np.random.seed(seed)
            pick_idx = np.random.choice(len(points), 1)
            # print(data_list[list], len(points), pick_idx)
            center_point = points[pick_idx, :].reshape(1, -1)
            select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]

            if sampling_way == "random":
                st = time.time()
                select_idx = DP.shuffle_idx(select_idx)
                select_points = points[select_idx]
                for sub in a[:start]:
                    select_points = select_points[0:sub]
                ed = time.time() - st
                test.append(ed)
                if list==0: print('check', select_points.shape)
            elif sampling_way == "fps":
                fps_xyz = points[select_idx]
                st = time.time()
                for sub in a[:start]:
                    fps_xyz = farthest_point_sample(fps_xyz, sub)
                ed = time.time() - st
                test.append(ed)
                if list==0: print('check', fps_xyz.shape)

            elif sampling_way == "polar":
                select_points = points[select_idx]
                grid = [64, 64, 16]
                st = time.time()
                select_points = polar_samplr(select_points, grid)
                for sub in a[:start]:
                    select_points = select_points[0:sub]
                ed = time.time() - st
                test.append(ed)
                if list==0: print('check', select_points.shape)

        print(sampling_way)
        print(a[:start])
        print("Mean time:{}\t std:{}".format(np.mean(test), np.std(test)))
    start += 1


