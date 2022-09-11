import yaml
import os
import torch
import time
import numpy as np
import torch.nn as nn
import shutil
from tqdm import tqdm

EXTENSIONS_SCAN = ['.npy']
def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

root = '/home/chx/Work/SemanticPOSS_dataset_grid_0.01/sequences' # dataset
DATA = yaml.safe_load(open("../utils/semantic-poss.yaml", 'r'))

lidar_list = []

sequences = DATA["split"]["train"]
for seq in sequences:
    # to string
    seq = '{0:02d}'.format(int(seq))
    print("parsing seq {}".format(seq))

    # get paths for each
    scan_path = os.path.join(root, seq, "velodyne")
    # get files
    print(scan_path)
    scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
    # extend list


    lidar_list.extend(scan_files)

# sort for correspondance
lidar_list.sort()

# bound = np.arange(0, 20)
bound = np.arange(0, 200)
DISTANCES = np.append(bound, [200, 10**3])
numbers = [0]*(len(DISTANCES)-1)
print(DISTANCES, len(DISTANCES))
for i in tqdm(range(len(lidar_list))):
    filename = lidar_list[i]
    # print(filename)
    scan = np.load(filename)
    scan = scan.reshape((-1, 3))
    points = scan[:, 0:3]  # get xyz
    depth = np.linalg.norm(points, 2, axis=1)

    for idx in range(len(DISTANCES)-1):
        # select by range
        lrange = DISTANCES[idx]
        hrange = DISTANCES[idx+1]

        # print(lrange, hrange)
        mask = np.logical_and(depth > lrange, depth <= hrange)
        nums = len(depth[mask])
        numbers[idx] += nums
np.save("d.npy", numbers)
print(numbers)

