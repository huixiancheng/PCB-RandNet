import os
import logging
import torch
import numpy as np

def seed_torch(seed=1024):
    import random
    import numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = False
    print("We use the seed: {}".format(seed))

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_logger(filename, verbosity=0, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", '%Y%m%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def copyFiles(target_path):
    import shutil
    target_path = target_path + 'code' + "/"
    if not os.path.isdir(target_path):
        os.makedirs(target_path)
    shutil.copyfile("train_SemanticKITTI.py", target_path + "train_SemanticKITTI.py")
    shutil.copyfile("train_both_SemanticKITTI.py", target_path + "train_both_SemanticKITTI.py")
    shutil.copyfile("train_SemanticPOSS.py", target_path + "train_SemanticPOSS.py")
    shutil.copyfile("train_both_SemanticPOSS.py", target_path + "train_both_SemanticPOSS.py")
    shutil.copyfile("test_SemanticKITTI.py", target_path + "test_SemanticKITTI.py")
    shutil.copyfile("test_SemanticPOSS.py", target_path + "test_SemanticPOSS.py")

    shutil.copyfile("config.py", target_path + "/config.py")
    # shutil.copyfile("network/ResNet.py", target_path + "/ResNet.py")

    temp_path = target_path + '/' + 'dataset'
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)
    shutil.rmtree(temp_path)
    shutil.copytree('dataset', temp_path)

    temp_path = target_path + '/' + 'network'
    if not os.path.isdir(temp_path):
        os.makedirs(temp_path)
    shutil.rmtree(temp_path)
    shutil.copytree('network', temp_path)