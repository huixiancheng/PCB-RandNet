from utils.data_process import DataProcessing as DP
from config import ConfigSemanticKITTI as cfg
from os.path import join
import numpy as np
import pickle
import torch.utils.data as torch_data
import torch

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)

def invperm(p):
    q = np.empty_like(p)
    q[p] = np.arange(len(p))
    return q

def find_map_2d(arr1, arr2):
    o1 = np.lexsort(arr1.T)
    o2 = np.lexsort(arr2.T)
    return o2[invperm(o1)]

class SemanticKITTI(torch_data.Dataset):
    def __init__(self, mode, sampling_way="random", step=0, grid=[64, 64, 16]):
        self.name = 'SemanticKITTI'
        self.dataset_path = '/home/chx/Work/semantic-kitti-randla/sequences'
        # self.dataset_path = '/home/chx/Work/semantic-kitti-sub/sequences'

        self.num_classes = cfg.num_classes
        self.ignored_labels = np.sort([0])

        self.mode = mode
        self.sampling_way = sampling_way

        self.grid = grid

        if mode == 'training':
            seq_list = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
            self.data_list = sorted(self.data_list)
            if step != 0:
                self.data_list = self.data_list[0:len(self.data_list):step]
        elif mode == 'validation':
            seq_list = ['08']
            self.data_list = DP.get_file_list(self.dataset_path, seq_list)
            self.data_list = sorted(self.data_list)
            if step != 0:
                self.data_list = self.data_list[0:len(self.data_list):step]
        print("Dataset path: " + self.dataset_path)
        print("Using {} scans from sequences {}".format(len(self.data_list), seq_list))
        print("Sampling Way :", self.sampling_way)
        if self.sampling_way == "polar":
            print("Grid: ", grid)

    def get_class_weight(self):
        return DP.get_class_weights(self.dataset_path, self.data_list, self.num_classes)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.data_list)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def spatially_regular_gen(self, item, data_list):
        # Generator loop
        cloud_ind = item
        pc_path = data_list[cloud_ind]
        pc, tree, labels = self.get_data(pc_path)
        # crop a small point cloud
        pick_idx = np.random.choice(len(pc), 1)
        selected_pc, selected_labels, selected_idx = self.crop_pc(pc, labels, tree, pick_idx)

        return selected_pc, selected_labels, selected_idx, np.array([cloud_ind], dtype=np.int32)

    def get_data(self, file_path):
        seq_id = file_path[0]
        frame_id = file_path[1]
        kd_tree_path = join(self.dataset_path, seq_id, 'KDTree', frame_id + '.pkl')
        # read pkl with search tree
        with open(kd_tree_path, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)
        # load labels
        label_path = join(self.dataset_path, seq_id, 'labels', frame_id + '.npy')
        labels = np.squeeze(np.load(label_path))
        return points, search_tree, labels

    # @staticmethod
    def crop_pc(self, points, labels, search_tree, pick_idx):
        # crop a fixed size point cloud for training
        center_point = points[pick_idx, :].reshape(1, -1)
        select_idx = search_tree.query(center_point, k=cfg.num_points)[1][0]

        if self.sampling_way == "random":
            select_idx = DP.shuffle_idx(select_idx)
            select_points = points[select_idx]
            select_labels = labels[select_idx]

        elif self.sampling_way == "polar":
            select_idx = DP.shuffle_idx(select_idx)
            random_points = temp_xyz = points[select_idx]
            random_label = temp_gt = labels[select_idx]

            grid = self.grid
            left_d_xyz, right_d_xyz, left_d_gt, right_d_gt = self.polar_samplr(temp_xyz, temp_gt, grid)

            select_points = np.concatenate((left_d_xyz, right_d_xyz), axis=0)
            select_labels = np.concatenate((left_d_gt, right_d_gt), axis=0)
            idx = find_map_2d(select_points, random_points)
            select_idx = select_idx[idx]

            # print(points)
            # print(select_points)
            # print(points[select_idx])
            # print((select_points==points[select_idx]).all())

        else:
            raise TypeError("Choose what type you want to sampling")

        return select_points, select_labels, select_idx

    @staticmethod
    def polar_samplr(mid_xyz, mid_gt, grid, fixed_volume_space=False):
        xyz_pol = cart2polar(mid_xyz)

        # print(xyz_pol, max(xyz_pol[:, 0]), max(xyz_pol[:, 1]), max(xyz_pol[:, 0]))
        if fixed_volume_space:
            max_volume_space, min_volume_space = [50, np.pi, 1.5], [0, -np.pi, -3]
            max_bound = np.asarray(max_volume_space)
            min_bound = np.asarray(min_volume_space)
        else:
            # max_bound_r = min(np.percentile(xyz_pol[:, 0], 100, axis=0), 50)
            max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
            min_bound_r = max(np.percentile(xyz_pol[:, 0], 0, axis=0), 3)
            max_bound_p = np.max(xyz_pol[:, 1], axis=0)
            min_bound_p = np.min(xyz_pol[:, 1], axis=0)

            max_bound_z = min(np.max(xyz_pol[:, 2], axis=0), 1.5)
            min_bound_z = max(np.min(xyz_pol[:, 2], axis=0), -3)

            max_bound = np.concatenate(([max_bound_r], [max_bound_p], [max_bound_z]))
            min_bound = np.concatenate(([min_bound_r], [min_bound_p], [min_bound_z]))
        cur_grid_size = np.asarray(grid)

        crop_range = max_bound - min_bound
        intervals = crop_range/(cur_grid_size-1)

        if (intervals==0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)


        keys, revers, counts = np.unique(grid_ind, return_inverse=True, return_counts=True, axis=0)


        idx = np.argsort(revers)
        # revers = revers[idx]
        mid_xyz = mid_xyz[idx]
        mid_gt = mid_gt[idx]
        slic = counts.cumsum()
        slic = np.insert(slic, 0, 0)

        left_xyz, right_xyz = np.zeros([0, 3]), np.zeros([0, 3])
        left_label, right_label = np.array([]), np.array([])

        small = counts[counts < 4]
        new_nums = len(mid_xyz) // 4 - sum(small)
        # new_nums = len(mid_xyz) // 4 - len(small)
        new_grid = len(counts) - len(small)
        sample_list = []
        for i in range(new_grid):
            curr = new_nums // new_grid
            sample_list.append(curr)
            new_nums -= curr
            new_grid -= 1
        sample_list = np.array(sample_list)
        sample_list = DP.shuffle_idx(sample_list)
        # print(sum(sample_list), sample_list, np.unique(sample_list))

        idx = 0
        for i in range(len(counts)):
            select_xyz = mid_xyz[slic[i]:slic[i+1]]
            select_labels = mid_gt[slic[i]:slic[i+1]]
            select_xyz, select_labels = DP.same_shuffl(select_xyz, select_labels)
            nubs = counts[i]
            if nubs >= 4:
                downs_n = sample_list[idx]
                idx += 1
                left_xyz = np.concatenate((left_xyz, select_xyz[0:downs_n]), axis=0)
                right_xyz = np.concatenate((right_xyz, select_xyz[downs_n:]), axis=0)
                left_label = np.concatenate((left_label, select_labels[0:downs_n]), axis=0)
                right_label = np.concatenate((right_label, select_labels[downs_n:]), axis=0)
            else:
                left_xyz = np.concatenate((left_xyz, select_xyz), axis=0)
                left_label = np.concatenate((left_label, select_labels), axis=0)
                # downs_n = 1
                # left_xyz = np.concatenate((left_xyz, select_xyz[0:downs_n]), axis=0)
                # right_xyz = np.concatenate((right_xyz, select_xyz[downs_n:]), axis=0)
                # left_label = np.concatenate((left_label, select_labels[0:downs_n]), axis=0)
                # right_label = np.concatenate((right_label, select_labels[downs_n:]), axis=0)

        supp = len(mid_xyz) // 4 - len(left_xyz)
        # print(supp)
        if supp == 0:
            pass
        elif supp > 0:
            right_xyz, right_label = DP.same_shuffl(right_xyz, right_label)
            left_xyz = np.concatenate((left_xyz, right_xyz[0:supp]))
            left_label = np.concatenate((left_label, right_label[0:supp]))
            right_xyz = right_xyz[supp:]
            right_label = right_label[supp:]
        else:
            left_xyz, left_label = DP.same_shuffl(left_xyz, left_label)
            supp = len(mid_xyz) // 4
            left_xyz, supp_xyz = left_xyz[:supp], left_xyz[supp:]
            left_label, supp_label = left_label[:supp], left_label[supp:]
            right_xyz = np.concatenate((supp_xyz, right_xyz))
            right_label = np.concatenate((supp_label, right_label))

        left_xyz, left_label = DP.same_shuffl(left_xyz, left_label)
        right_xyz, right_label = DP.same_shuffl(right_xyz, right_label)

        return left_xyz, right_xyz, left_label, right_label

    def tf_map(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]
        return input_list

    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs = self.tf_map(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        return inputs

    def tf_map_baf_lac(self, batch_pc, batch_label, batch_pc_idx, batch_cloud_idx):
        features = batch_pc
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers):
            neighbour_idx = DP.knn_search(batch_pc, batch_pc, cfg.k_n)
            sub_points = batch_pc[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            pool_i = neighbour_idx[:, :batch_pc.shape[1] // cfg.sub_sampling_ratio[i], :]
            up_i = DP.knn_search(sub_points, batch_pc, 1)
            input_points.append(batch_pc)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_pc = sub_points

        batch_graph_idx = DP.knn_search(input_points[0], input_points[0], cfg.k_n)
        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [features, batch_label, batch_pc_idx, batch_cloud_idx]
        return input_list, batch_graph_idx


    def collate_fn_baf_lac(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        flat_inputs, batch_graph_idx = self.tf_map_baf_lac(selected_pc, selected_labels, selected_idx, cloud_ind)

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1, 2).float()
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()

        inputs['graph_idx'] = torch.from_numpy(batch_graph_idx).long()
        return inputs

if __name__ == "__main__":

    import random
    import os

    def seed_torch(seed=1111):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False
        print("We use the seed: {}".format(seed))
    seed_torch(233)


    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    from torch.utils.data import DataLoader
    from tqdm import tqdm

    train_dataset = SemanticKITTI('training', "random")
    random_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=my_worker_init_fn,
        collate_fn=train_dataset.collate_fn,
        pin_memory=False
    )

    train_dataset = SemanticKITTI('training', "polar")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        worker_init_fn=my_worker_init_fn,
        collate_fn=train_dataset.collate_fn,
        pin_memory=False
    )

    tqdm_loader = tqdm(train_loader, total=len(train_loader))

    for epoch in range(100):
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx ==10:
                exit()
