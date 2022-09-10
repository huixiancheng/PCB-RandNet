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
from torch.utils.tensorboard import SummaryWriter
# my module
from dataset.semkitti_trainset import SemanticKITTI

import torch.nn.functional as F
from network.loss_func import compute_loss
from utils.metric import compute_acc, IoUCalculator, iouEval
from help_utils import seed_torch, my_worker_init_fn, get_logger, copyFiles, AverageMeter #seed_torch, my_worker_init_fn,

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='randla', choices=['randla', 'baflac', 'baaf'])
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', type=str, default='base', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=8, help='Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=8, help='Number of workers [default: 5]')
parser.add_argument('--sampling', type=str, default='random', choices=['random', 'polar'], help='Polar sample or not')
parser.add_argument('--seed', type=int, default=1024, help='Random Seed')
parser.add_argument('--step', type=int, default=4, help='sub dataset size')
parser.add_argument('--grid', nargs='+', type=int, default=[64, 64, 16], help='grid size of BEV representation')
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
        save_path = './save_semantic/' + FLAGS.log_dir + '/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        copyFiles(save_path)
        self.log_dir = save_path
        log_fname = os.path.join(self.log_dir, 'log_train.txt')
        self.logger = get_logger(log_fname, name="Trainer")

        argsDict = FLAGS.__dict__
        for eachArg, value in argsDict.items():
            self.logger.info(eachArg + ' : ' + str(value))

        train_dataset = SemanticKITTI('training', sampling_way=FLAGS.sampling, step=FLAGS.step, grid=FLAGS.grid)
        val_dataset = SemanticKITTI('validation', sampling_way=FLAGS.sampling, step=FLAGS.step, grid=FLAGS.grid)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if FLAGS.backbone == 'baflac':
            from network.BAF_LAC import BAF_LAC
            self.logger.info("Use Baseline: BAF-LAC")
            self.net = BAF_LAC(cfg)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn_baf_lac

        elif FLAGS.backbone == 'randla':
            from network.RandLANet import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn

        elif FLAGS.backbone == 'baaf':
            from network.BAAF import Network
            self.logger.info("Use Baseline: BAAF")
            self.net = Network(cfg)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn

        else:
            raise TypeError("1~5~!! can can need !!!")

        self.train_loader = DataLoader(
            train_dataset, batch_size=FLAGS.batch_size,
            shuffle=True, num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
        self.val_loader = DataLoader(
            val_dataset, batch_size=FLAGS.val_batch_size,
            shuffle=False, num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)

        self.logger.info((str(self.net)))
        pytorch_total_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        self.logger.info("Number of parameters: {} ".format(pytorch_total_params / 1000000) + "M")

        # Load the Adam optimizer
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, 0.95)

        # Load module
        self.highest_val_iou = 0
        self.start_epoch = 0
        CHECKPOINT_PATH = FLAGS.checkpoint_path
        if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
            self.logger.info("Load Pretrain")
            checkpoint = torch.load(CHECKPOINT_PATH)
            self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']

        # Loss Function
        class_weights = torch.tensor([[0, 17.1782, 49.4506, 49.0822, 45.9189, 44.9322, 49.0659, 49.6848, 49.8643,
          5.3644, 31.3474,  7.2694, 41.0078,  5.5935, 11.1378,  2.8731, 37.3568,
          9.1691, 43.3190, 48.0684]]).cuda()
        # class_weights = train_dataset.get_class_weight()
        self.logger.info(class_weights)
        # class_weights = torch.from_numpy(class_weights).float().cuda()
        # self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.evaluator = iouEval(20, self.device, 0)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        losses = AverageMeter()
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))

        scaler = torch.cuda.amp.GradScaler()
        for batch_idx, batch_data in enumerate(tqdm_loader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(cfg.num_layers):
                        batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                else:
                    batch_data[key] = batch_data[key].cuda(non_blocking=True)

            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                semantic_out = self.net(batch_data)
                loss = self.criterion(semantic_out, batch_data['labels']).mean()
                # loss, end_points = compute_loss(semantic_out, batch_data, self.train_dataset, self.criterion)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            # loss.backward()
            # self.optimizer.step()

            losses.update(loss.item())

            if batch_idx % 50 == 0 or batch_idx == len(tqdm_loader)-1:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info('Step {:08d} || Lr={:.6f} || L_out={loss.val:.4f}/({loss.avg:.4f})'.format(batch_idx, lr, loss=losses))
        self.scheduler.step()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))
            self.train_one_epoch()
            self.logger.info('**** EVAL EPOCH %03d ****' % (epoch))
            mean_iou = self.validate()
            # Save best checkpoint
            if mean_iou > self.highest_val_iou:
                self.logger.info('**** Current: %03f Best: %03f ****' % (mean_iou, self.highest_val_iou))
                self.highest_val_iou = mean_iou
                checkpoint_file = os.path.join(self.log_dir, 'checkpoint-best.tar')
                self.save_checkpoint(checkpoint_file)
            else:
                self.logger.info('**** Current: %03f Best: %03f ****' % (mean_iou, self.highest_val_iou))
            checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
            self.save_checkpoint(checkpoint_file)

    def validate(self):
        # torch.cuda.empty_cache()
        self.net.eval()  # set model to eval mode (for bn and dp)
        self.evaluator.reset()
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm_loader):
                for key in batch_data:
                    if type(batch_data[key]) is list:
                        for i in range(cfg.num_layers):
                            batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
                    else:
                        batch_data[key] = batch_data[key].cuda(non_blocking=True)

                # Forward pass
                # torch.cuda.synchronize()
                semantic_out = self.net(batch_data)

                argmax = F.softmax(semantic_out, dim=1).argmax(dim=1)
                self.evaluator.addBatch(argmax, batch_data['labels'])
                # loss, end_points = compute_loss(semantic_out, batch_data, self.train_dataset, self.criterion)
                # acc, end_points = compute_acc(end_points)
                # iou_calc.add_data(end_points)

        # mean_iou, iou_list = self.evaluator.getIoU()
        # mean_iou, iou_list = iou_calc.compute_iou()
        # self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        # s = 'IoU:'
        # for iou_tmp in iou_list:
        #     s += '{:5.2f} '.format(100 * iou_tmp)
        # self.logger.info(s)

        accuracy = self.evaluator.getacc()
        mean_iou, class_jaccard = self.evaluator.getIoU()
        class_func = ["unlabeled", "car", "bicycle", "motorcycle", "truck",
                      "other-vehicle", "person", "bicyclist", "motorcyclist", "road",
                      "parking", "sidewalk", "other-ground", "building", "fence",
                      "vegetation", "trunk", "terrain", "pole", "traffic-sign"]
        self.logger.info('Validation set: ||' 'Acc avg {acc:.3f} ||' 'IoU avg {iou:.3f}'.format(acc=accuracy, iou=mean_iou))
        for i, jacc in enumerate(class_jaccard):
            self.logger.info('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(i=i, class_str=class_func[i], jacc=jacc))
        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
