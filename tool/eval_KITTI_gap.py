# Common
import os
import logging
import warnings
import argparse
import numpy as np
from tqdm import tqdm
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.two_semkitti_trainset import SemanticKITTI
from utils.metric import compute_acc, IoUCalculator, iouEval

import torch.nn.functional as F
from help_utils import seed_torch, my_worker_init_fn, get_logger, copyFiles, AverageMeter

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='randla', choices=['randla', 'baflac', 'baaf'])
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', type=str, default='GAP+TEST', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--val_batch_size', type=int, default=1, help='Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=0, help='Number of workers [default: 5]')
parser.add_argument('--seed', type=int, default=1024, help='Polar sample or not')
parser.add_argument('--step', type=int, default=0, help='sub dataset size')
FLAGS = parser.parse_args()

seed_torch(FLAGS.seed)
torch.backends.cudnn.enabled = False

if FLAGS.backbone == 'baflac':
    from config import ConfigSemanticKITTI_BAF as cfg
else:
    from config import ConfigSemanticKITTI as cfg

class Trainer:
    def __init__(self):
        # Init Logging
        save_path = './save_semantic_poss/' + FLAGS.log_dir + '/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        copyFiles(save_path)
        self.log_dir = save_path
        log_fname = os.path.join(self.log_dir, 'log_train.txt')
        self.logger = get_logger(log_fname, name='Train')

        argsDict = FLAGS.__dict__
        for eachArg, value in argsDict.items():
            self.logger.info(eachArg + ' : ' + str(value))

        # tensorboard writer
        val_dataset = SemanticKITTI('validation', step=FLAGS.step)

        # Network & Optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if FLAGS.backbone == 'baflac':
            from network.BAF_LAC import BAF_LAC
            self.logger.info("Use Baseline: BAF-LAC")
            self.net = BAF_LAC(cfg)
            self.net.to(self.device)
            collate_fn = val_dataset.collate_fn_baf_lac

        elif FLAGS.backbone == 'randla':
            from network.RandLANet import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg)
            self.net.to(self.device)
            collate_fn = val_dataset.collate_fn

        elif FLAGS.backbone == 'baaf':
            from network.BAAF import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg)
            self.net.to(self.device)
            collate_fn = val_dataset.collate_fn

        else:
            raise TypeError("1~5~!! can can need !!!")

        self.val_loader = DataLoader(
            val_dataset, batch_size=FLAGS.val_batch_size,
            shuffle=False, num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=False)

        # self.logger.info((str(self.net)))
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.logger.info("Number of parameters: {} ".format(pytorch_total_params / 1000000) + "M")

        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            self.logger.info("Load Pretrain from " + CHECKPOINT_PATH)
            checkpoint = torch.load(CHECKPOINT_PATH)
            try:
                self.logger.info("Loading strict=True")
                self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            except:
                self.logger.info("Loading strict=False")
                self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            self.start_epoch = checkpoint['epoch']

        # Loss Function
        self.val_dataset = val_dataset

        self.evaluator = iouEval(20, self.device, 0)
        self.evaluator2 = iouEval(20, self.device, 0)

    def train(self):
        self.net.eval()  # set model to eval mode (for bn and dp)
        # iou_calc_polar = IoUCalculator(cfg)
        # iou_calc_random = IoUCalculator(cfg)
        self.evaluator.reset()
        self.evaluator2.reset()
        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            cnn = []
            cnn2 = []
            for batch_idx, (polar_data, random_data, idx) in enumerate(tqdm_loader):
                for key in polar_data:
                    if type(polar_data[key]) is list:
                        for i in range(cfg.num_layers):
                            polar_data[key][i] = polar_data[key][i].cuda(non_blocking=True)
                    else:
                        polar_data[key] = polar_data[key].cuda(non_blocking=True)
                for key in random_data:
                    if type(random_data[key]) is list:
                        for i in range(cfg.num_layers):
                            random_data[key][i] = random_data[key][i].cuda(non_blocking=True)
                    else:
                        random_data[key] = random_data[key].cuda(non_blocking=True)
                import time
                st = time.time()
                semantic_out = self.net(polar_data)
                torch.cuda.synchronize()
                ed = time.time() - st
                cnn.append(ed)
                print(ed)
                argmax = F.softmax(semantic_out, dim=1).argmax(dim=1)
                self.evaluator.addBatch(argmax, polar_data['labels'])

                # end_points = self.get_pred(semantic_out, polar_data, self.val_dataset)
                # acc, end_points = compute_acc(end_points)
                # iou_calc_polar.add_data(end_points)
                st = time.time()
                semantic_out = self.net(random_data)
                torch.cuda.synchronize()
                ed = time.time() - st
                cnn2.append(ed)
                print(ed)
                argmax = F.softmax(semantic_out, dim=1).argmax(dim=1)
                self.evaluator2.addBatch(argmax, random_data['labels'])
                # end_points = self.get_pred(semantic_out, random_data, self.val_dataset)
                # acc, end_points = compute_acc(end_points)
                # iou_calc_random.add_data(end_points)
                if batch_idx==50:
                    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
                    print("Mean KNN inference time:{}\t std:{}".format(np.mean(cnn2), np.std(cnn2)))
                    exit()
        # mean_iou, iou_list = iou_calc_polar.compute_iou()
        mean_iou, iou_list = self.evaluator.getIoU()
        self.logger.info('mean IoU:{:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)

        # mean_iou, iou_list = iou_calc_random.compute_iou()
        mean_iou, iou_list = self.evaluator2.getIoU()
        self.logger.info('mean IoU:{:.2f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)

    @staticmethod
    def get_pred(semantic_out, end_points, dataset):
        logits = semantic_out
        labels = end_points['labels']
        end_points = {}
        logits = logits.transpose(1, 2).reshape(-1, dataset.num_classes)
        labels = labels.reshape(-1)

        # Boolean mask of points that should be ignored
        ignored_bool = (labels == 0)

        for ign_label in dataset.ignored_labels:
            ignored_bool = ignored_bool | (labels == ign_label)

        # Collect logits and labels that are not ignored
        valid_idx = ignored_bool == 0
        valid_logits = logits[valid_idx, :]
        valid_labels_init = labels[valid_idx]
        # Reduce label values in the range of logit shape
        reducing_list = torch.arange(0, dataset.num_classes).long().to(logits.device)
        inserted_value = torch.zeros((1,)).long().to(logits.device)
        for ign_label in dataset.ignored_labels:
            reducing_list = torch.cat([reducing_list[:ign_label], inserted_value, reducing_list[ign_label:]], 0)
        valid_labels = torch.gather(reducing_list, 0, valid_labels_init)
        end_points['valid_logits'], end_points['valid_labels'] = valid_logits, valid_labels
        return end_points
def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
