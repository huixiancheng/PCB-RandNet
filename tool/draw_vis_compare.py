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
sampling_way = 'polar'

if mode == 'training':
    seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
    data_list = DP.get_file_list(dataset_path, seq_list)
    data_list = sorted(data_list)
elif mode == 'validation':
    seq_list = ['08']
    data_list = DP.get_file_list(dataset_path, seq_list)
    data_list = sorted(data_list)

# if mode == 'training':
#     seq_list = ['00', '01', '03', '04', '05']
#     data_list = DP.get_file_list(dataset_path, seq_list)
#     data_list = sorted(data_list)
# elif mode == 'validation':
#     seq_list = ['02']
#     data_list = DP.get_file_list(dataset_path, seq_list)
#     data_list = sorted(data_list)

print("Using {} scans from sequences {}".format(len(data_list), seq_list))
print("Sampling Way :", sampling_way)


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

def polar_samplr(mid_xyz, grid, temp=0, fixed_volume_space=False):
    xyz_pol = cart2polar(mid_xyz)

    # print(xyz_pol, max(xyz_pol[:, 0]), max(xyz_pol[:, 1]), max(xyz_pol[:, 0]))
    # if fixed_volume_space:
    #     max_volume_space, min_volume_space = [50, np.pi, 1.5], [0, -np.pi, -3]
    #     max_bound = np.asarray(max_volume_space)
    #     min_bound = np.asarray(min_volume_space)
    # else:
    #     max_bound_r = min(np.percentile(xyz_pol[:, 0], 100, axis=0), 50)
    #     min_bound_r = max(np.percentile(xyz_pol[:, 0], 0, axis=0), 3)
    #     max_bound_p = np.max(xyz_pol[:, 1], axis=0)
    #     min_bound_p = np.min(xyz_pol[:, 1], axis=0)
    #
    #     max_bound_z = min(np.max(xyz_pol[:, 2], axis=0), 1.5)
    #     min_bound_z = max(np.min(xyz_pol[:, 2], axis=0), -3)
    #
    #     max_bound = np.concatenate(([max_bound_r], [max_bound_p], [max_bound_z]))
    #     min_bound = np.concatenate(([min_bound_r], [min_bound_p], [min_bound_z]))
    # cur_grid_size = np.asarray(grid)

    if fixed_volume_space:
        max_volume_space, min_volume_space = [80, np.pi, 3], [0, -np.pi, -3]
        max_bound = np.asarray(max_volume_space)
        min_bound = np.asarray(min_volume_space)
    else:
        # max_bound_r = min(np.percentile(xyz_pol[:, 0], 100, axis=0), 80)
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = max(np.percentile(xyz_pol[:, 0], 0, axis=0), 3)
        max_bound_p = np.max(xyz_pol[:, 1], axis=0)
        min_bound_p = np.min(xyz_pol[:, 1], axis=0)

        max_bound_z = min(np.max(xyz_pol[:, 2], axis=0), 3)
        min_bound_z = max(np.min(xyz_pol[:, 2], axis=0), -3)

        max_bound = np.concatenate(([max_bound_r], [max_bound_p], [max_bound_z]))
        min_bound = np.concatenate(([min_bound_r], [min_bound_p], [min_bound_z]))
    cur_grid_size = np.asarray(grid)

    print(max_bound, min_bound)
    crop_range = max_bound - min_bound
    intervals = crop_range/(cur_grid_size-1)

    if (intervals==0).any(): print("Zero interval!")

    grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)
    keys, revers, counts = np.unique(grid_ind, return_inverse=True, return_counts=True, axis=0)

    import matplotlib.pyplot as plt
    counts = np.sort(counts)[::-1]
    fig = plt.figure(figsize=(4, 4), facecolor='w', dpi=100)
    print(counts)
    for i in range(len(counts)):
        plt.bar(i, counts[i])
    # plt.show()
    plt.xlabel('Nums of Cylindrical Block')
    plt.ylabel('Nums of Points')
    plt.axhline(y=64, ls="-", c='r', label='N = 64', linewidth=1.0)
    plt.axhline(y=16, ls="-", c='b', label='N = 16', linewidth=1.0)
    plt.axhline(y=8, ls="-", c='c', label='N = 8', linewidth=1.0)
    plt.axhline(y=4, ls="-", c='m', label='N = 4', linewidth=1.0)
    plt.legend()
    fig.set_size_inches(4, 4, forward=True)
    fig.tight_layout()
    plt.savefig("pic/{}.png".format(temp))
    # plt.show()
    plt.close()



    # return mid_xyz, mid_gt
    idx = np.argsort(revers)
    # revers = revers[idx]
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
    print(len(counts), np.unique(sample_list))
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
    print(supp)
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
    np.random.shuffle(right_xyz)
    select_points = np.concatenate((left_xyz, right_xyz), axis=0)

    return select_points

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
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

import random
random.seed(46441664)
random.shuffle(data_list)

def draw(fig, points, a, b, c, max_r=50):
    bound = np.arange(0, max_r)
    DISTANCES = np.append(bound, [max_r, 10 ** 3])
    depth = np.linalg.norm(points, 2, axis=1)
    numbers = [0] * (len(DISTANCES)-1)
    for idx in range(len(DISTANCES) - 1):
        # select by range
        lrange = DISTANCES[idx]
        hrange = DISTANCES[idx + 1]

        mask = np.logical_and(depth > lrange, depth <= hrange)
        nums = len(depth[mask])
        numbers[idx] += nums

    # fig = plt.figure(figsize=(24, 4), facecolor='w', dpi=100)
    ax = fig.add_subplot(a, b, c)

    for i in range(len(numbers)):
        # print(numbers[i])
        ax.bar(i, numbers[i])

    ax.set_xlabel('Range of Distance to Lidar Sensor')
    ax.set_ylabel('Number of Points')

# def draw(fig, points, a, b, c, max_r=50):
#     bound = np.arange(0, max_r, 10)
#     DISTANCES = np.append(bound, [max_r, 10 ** 3])
#     depth = np.linalg.norm(points, 2, axis=1)
#     numbers = [0] * (len(DISTANCES)-1)
#     for idx in range(len(DISTANCES) - 1):
#         # select by range
#         lrange = DISTANCES[idx]
#         hrange = DISTANCES[idx + 1]
#
#         mask = np.logical_and(depth > lrange, depth <= hrange)
#         nums = len(depth[mask])
#         numbers[idx] += nums
#
#     ax = fig.add_subplot(a, b, c)
#     for i in range(len(numbers)):
#         ax.bar(i, numbers[i])
#     ax.set_xlabel('Range of Distance to Lidar Sensor')
#     ax.set_ylabel('Number of Points')

cp = []

# dian, xia = [11264, 2816, 704, 176], [4, 16, 64, 256]
dian, xia = [11264, 2816], [4, 16]

import os
save_path = 'pic'
if not (os.path.exists(save_path)):
    os.makedirs(save_path)

for list in tqdm(range(len(data_list))):
    if list == 10:
        exit()
    print(data_list[list])
    points, search_tree, labels = get_data(data_list[list])
    print(len(points))
    pick_idx = np.random.choice(len(points), 1)

    center_point = points[pick_idx, :].reshape(1, -1)

    center_range = np.linalg.norm(center_point, 2, axis=1)
    # cp.append(center_range)
    select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]

    if sampling_way == "random":

        st = time.time()
        select_idx = DP.shuffle_idx(select_idx)
        select_points = points[select_idx]
        select_labels = labels[select_idx]
        ed = time.time() - st


    elif sampling_way == "polar":
        # polar
        select_idx = DP.shuffle_idx(select_idx)
        mid_xyz = temp_xyz = points[select_idx]
        temp_gt = labels[select_idx]
        grid = [32, 32, 16]

        # select_points = polar_samplr(temp_xyz, grid, list)



        # fig = plt.figure(figsize=(30, 4), facecolor='w', dpi=100)
        # ax = fig.add_subplot(1, len(dian)+1, 1, projection='3d')
        # fig.subplots_adjust(wspace=0.2)
        # x, y, z = mid_xyz[:, 0], mid_xyz[:, 1], mid_xyz[:, 2]
        # ax.scatter(x, y, z, s=0.1)
        # ax.title.set_text('Original Points \n Nums: 45056')
        #
        # fig2 = plt.figure(figsize=(30, 4), facecolor='w', dpi=100)
        # fig2.subplots_adjust(wspace=0.2)
        # draw(fig2, mid_xyz, 1, len(dian)+1, 1)
        #
        # fps_xyz = mid_xyz
        # temp = 2
        # for i, j in zip(dian, xia):
        #     fps_xyz = farthest_point_sample(fps_xyz, i)  # [B, npoint, C]
        #     ax = fig.add_subplot(1, len(dian)+1, temp, projection='3d')
        #
        #     x, y, z = fps_xyz[:, 0], fps_xyz[:, 1], fps_xyz[:, 2]
        #     ax.scatter(x[0:i], y[0:i], z[0:i], s=0.1)
        #     ax.title.set_text('FPS Dowsampling to: 1 / %d \n Nums: %d' % (j, i))
        #     draw(fig2, fps_xyz[0:i], 1, len(dian)+1, temp)
        #     temp += 1
        # fig.set_size_inches(30, 4, forward=True)
        # fig.tight_layout()
        # fig.savefig("pic/fps_{}.png".format(list))
        # fig.show()
        # # fig.close()
        #
        # fig2.set_size_inches(30, 4, forward=True)
        # fig2.tight_layout()
        # fig2.savefig("pic/fps_dis_{}.png".format(list))
        # fig2.show()

        fig = plt.figure(figsize=(24, 4), facecolor='w', dpi=100)
        ax = fig.add_subplot(1, 5, 1, projection='3d')
        fig.subplots_adjust(wspace=0.3)

        x, y, z = mid_xyz[:, 0], mid_xyz[:, 1], mid_xyz[:, 2]
        ax.scatter(x, y, z, s=0.1)
        ax.title.set_text('Original Points \n Nums: 45056')

        fig2 = plt.figure(figsize=(24, 4), facecolor='w', dpi=100)
        fig2.subplots_adjust(wspace=0.3)
        draw(fig2, mid_xyz, 1, 5, 1)

        temp = 2
        for i, j in zip(dian, xia):
            ax = fig.add_subplot(1, 5, temp, projection='3d')
            x, y, z = mid_xyz[:, 0], mid_xyz[:, 1], mid_xyz[:, 2]
            ax.scatter(x[0:i], y[0:i], z[0:i], s=0.1)
            ax.title.set_text('Random Dowsampling to: 1 / %d \n Nums: %d' % (j, i))
            draw(fig2, mid_xyz[0:i], 1, 5, temp)
            temp += 1
        # fig.set_size_inches(14, 4, forward=True)
        # fig.tight_layout()
        # fig.savefig("pic/random_{}.png".format(list))
        # fig.show()
        # # fig.close()
        #
        # fig2.set_size_inches(14, 4, forward=True)
        # fig2.tight_layout()
        # fig2.savefig("pic/random_dis_{}.png".format(list))
        # fig2.show()
        # fig2.close()


        grid = [64, 64, 16]
        select_points = polar_samplr(temp_xyz, grid, temp=list)
        # fig = plt.figure(figsize=(14, 4), facecolor='w', dpi=100)
        # ax = fig.add_subplot(1, len(dian)+1, 1, projection='3d')
        # fig.subplots_adjust(wspace=0.3)
        # x, y, z = select_points[:, 0], select_points[:, 1], select_points[:, 2]
        # ax.scatter(x, y, z, s=0.1)
        # ax.title.set_text('Original Points \n Nums: 45056')

        # fig2 = plt.figure(figsize=(14, 4), facecolor='w', dpi=100)
        # fig2.subplots_adjust(wspace=0.3)
        # draw(fig2, mid_xyz, 1, len(dian)+1, 1)


        temp = 4
        for i, j in zip(dian, xia):
            ax = fig.add_subplot(1, 5, temp, projection='3d')
            x, y, z = select_points[:, 0], select_points[:, 1], select_points[:, 2]
            ax.scatter(x[0:i], y[0:i], z[0:i], s=0.1)
            ax.title.set_text('Polar Dowsampling to: 1 / %d \n Nums: %d' % (j, i))
            draw(fig2, select_points[0:i], 1, 5, temp)
            temp += 1

        line = plt.Line2D((.2, .2), (.1, .9), color="k", linewidth=3)
        fig.add_artist(line)
        line = plt.Line2D((.6, .6), (.1, .9), color="k", linewidth=3)
        fig.add_artist(line)

        fig.set_size_inches(24, 4, forward=True)
        fig.tight_layout()
        fig.savefig("pic/polar_{}.png".format(list))
        fig.show()
        # fig.close()

        # line = plt.Line2D((.2, .2), (.1, .9), color="k", linewidth=3)
        # fig2.add_artist(line)
        # line = plt.Line2D((.6, .6), (.1, .9), color="k", linewidth=3)
        # fig2.add_artist(line)
        fig2.set_size_inches(24, 4, forward=True)
        fig2.tight_layout()

        fig2.savefig("pic/polar_dis_{}.png".format(list))
        fig2.show()

        # exit()
# import matplotlib.pyplot as plt
#
# cp = np.array(cp)
# normal_data = cp.reshape(-1, 1).squeeze()
# weights = np.ones_like(normal_data) / float(len(normal_data))
# plt.hist(normal_data, bins=[0, 10, 20, 30, 40, 50], color='pink', edgecolor='k', weights=weights)
# plt.title('Distribution')
# plt.xlabel("Value of Distance")
# plt.ylabel("Frequency")
# plt.show()