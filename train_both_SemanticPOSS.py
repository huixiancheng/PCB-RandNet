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
# my module
from dataset.two_poss_trainset import SemanticPOSS
from network.loss_func import compute_loss
from utils.metric import compute_acc, IoUCalculator

import torch.nn.functional as F
from help_utils import seed_torch, my_worker_init_fn, get_logger, copyFiles, AverageMeter

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--backbone', type=str, default='randla', choices=['randla', 'baflac', 'baaf'])
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', type=str, default='randla-both', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=6, help='Batch Size during training [default: 5]')
parser.add_argument('--val_batch_size', type=int, default=8, help='Batch Size during training [default: 30]')
parser.add_argument('--num_workers', type=int, default=6, help='Number of workers [default: 5]')
parser.add_argument('--seed', type=int, default=1024, help='Polar sample or not')
parser.add_argument('--grid', nargs='+', type=int, default=[64, 64, 16], help='grid size of BEV representation')
FLAGS = parser.parse_args()

seed_torch(FLAGS.seed)
torch.backends.cudnn.enabled = False

if FLAGS.backbone == 'baflac':
    from config import ConfigSemanticPOSS_BAF as cfg
else:
    from config import ConfigSemanticPOSS as cfg

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
        train_dataset = SemanticPOSS('training', grid=FLAGS.grid)
        val_dataset = SemanticPOSS('validation', grid=FLAGS.grid)

        # Network & Optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if FLAGS.backbone == 'baflac':
            from network.BAF_LAC import BAF_LAC
            self.logger.info("Use Baseline: BAF-LAC")
            self.net = BAF_LAC(cfg, learn=True)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn_baf_lac

        elif FLAGS.backbone == 'randla':
            from network.RandLANet import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg, learn=True)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn

        elif FLAGS.backbone == 'baaf':
            from network.BAAF import Network
            self.logger.info("Use Baseline: Rand-LA")
            self.net = Network(cfg, learn=True)
            self.net.to(self.device)
            collate_fn = train_dataset.collate_fn

        else:
            raise TypeError("1~5~!! can can need !!!")

        self.train_loader = DataLoader(
            train_dataset, batch_size=FLAGS.batch_size,
            shuffle=True, num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=False)
        self.val_loader = DataLoader(
            val_dataset, batch_size=FLAGS.val_batch_size,
            shuffle=False, num_workers=FLAGS.num_workers,
            worker_init_fn=my_worker_init_fn, collate_fn=collate_fn, pin_memory=False)


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
        self.db_criterion = nn.L1Loss(reduction='mean')
        class_weights = train_dataset.get_class_weight()
        print(class_weights)
        class_weights = torch.from_numpy(class_weights).float().cuda()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def consistency_loss_l1(self, pred_cls, pred_cls_raw):
        '''
        Input:
        pred_cls, pred_cls_raw (BS, C, N, 1)
        '''
        pred_cls_softmax = F.softmax(pred_cls, dim=1)
        pred_cls_raw_softmax = F.softmax(pred_cls_raw, dim=1)
        loss = (pred_cls_softmax - pred_cls_raw_softmax).abs().sum(dim=1).mean()
        return loss

    def train_one_epoch(self):
        self.net.train()  # set model to training mode
        total_losses = AverageMeter()
        losses = AverageMeter()
        db_losses = AverageMeter()
        tqdm_loader = tqdm(self.train_loader, total=len(self.train_loader))

        scaler = torch.cuda.amp.GradScaler()

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

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                random_out, _ = self.net(random_data)
                polar_out, sigma = self.net(polar_data)
                # print(sigma)
                loss, end_points = compute_loss(polar_out, polar_data, self.train_dataset, self.criterion)

                idx = idx[:, None, :].cuda(non_blocking=True)
                random_out = torch.take_along_dim(random_out, indices=idx, dim=-1)
                db_loss = self.consistency_loss_l1(polar_out, random_out)

                factor_ce = 1.0 / (sigma[0] ** 2)
                factor_l1 = 1.0 / (sigma[1] ** 2)
                total_loss = factor_ce * loss + factor_l1 * db_loss + \
                       torch.log(1+sigma[0]) + \
                       torch.log(1+sigma[1])
                # total_loss = loss + 15*db_loss
            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # loss.backward()
            # self.optimizer.step()
            losses.update(loss.item())
            db_losses.update(db_loss.item())
            total_losses.update(total_loss.item())
            if batch_idx % 50 == 0 or batch_idx == len(tqdm_loader)-1:
                self.logger.info('{:.6f} || {:.6f} || {:.6f} || {:.6f}'.format(factor_ce.item() * loss.item(),
                                 factor_l1.item() * db_loss.item(), factor_ce.item(), factor_l1.item()))
                self.logger.info('{:.6f} || {:.6f} || {:.6f} || {:.6f}'.format(torch.log(1+sigma[0]).item(),
                                torch.log(1+sigma[1]).item(), sigma[0].item(), sigma[1].item()))

                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info('Step {:08d} || Lr={:.6f} || '
                                 'L_total={total.val:.4f}/({total.avg:.4f}) || '
                                 'L_ce={loss.val:.4f}/({loss.avg:.4f}) '
                                 '|| L_db={db.val:.4f}/({db.avg:.4f})'.format(batch_idx, lr, total=total_losses, loss=losses, db=db_losses))
                # exit()
        self.scheduler.step()

    def train(self):
        for epoch in range(self.start_epoch, FLAGS.max_epoch):
            self.cur_epoch = epoch
            self.logger.info('**** EPOCH %03d ****' % (epoch))

            self.train_one_epoch()
            checkpoint_file = os.path.join(self.log_dir, 'checkpoint.tar')
            self.save_checkpoint(checkpoint_file)
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

    def validate(self):
        # torch.cuda.empty_cache()
        self.net.eval()  # set model to eval mode (for bn and dp)
        iou_calc = IoUCalculator(cfg)

        tqdm_loader = tqdm(self.val_loader, total=len(self.val_loader))
        with torch.no_grad():
            for batch_idx, (polar_data, random_data, idx) in enumerate(tqdm_loader):
                for key in polar_data:
                    if type(polar_data[key]) is list:
                        for i in range(cfg.num_layers):
                            polar_data[key][i] = polar_data[key][i].cuda(non_blocking=True)
                    else:
                        polar_data[key] = polar_data[key].cuda(non_blocking=True)
                # for key in random_data:
                #     if type(random_data[key]) is list:
                #         for i in range(cfg.num_layers):
                #             random_data[key][i] = random_data[key][i].cuda(non_blocking=True)
                #     else:
                #         random_data[key] = random_data[key].cuda(non_blocking=True)
                # Forward pass
                # torch.cuda.synchronize()

                semantic_out, _ = self.net(polar_data)

                loss, end_points = compute_loss(semantic_out, polar_data, self.train_dataset, self.criterion)
                acc, end_points = compute_acc(end_points)
                iou_calc.add_data(end_points)

        mean_iou, iou_list = iou_calc.compute_iou()
        self.logger.info('mean IoU:{:.1f}'.format(mean_iou * 100))
        s = 'IoU:'
        for iou_tmp in iou_list:
            s += '{:5.2f} '.format(100 * iou_tmp)
        self.logger.info(s)

        return mean_iou

    def save_checkpoint(self, fname):
        save_dict = {
            'epoch': self.cur_epoch+1,  # after training one epoch, the start_epoch should be epoch+1
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        # with nn.DataParallel() the net is added as a submodule of DataParallel
        try:
            save_dict['model_state_dict'] = self.net.module.state_dict()
        except AttributeError:
            save_dict['model_state_dict'] = self.net.state_dict()
        torch.save(save_dict, fname)


def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
