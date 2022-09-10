import torch
import torch.nn as nn
import torch.nn.functional as F
import network.pytorch_utils as pt_utils


class Network(nn.Module):

    def __init__(self, config, learn=False):
        super().__init__()
        self.config = config
        self.learn = learn
        self.fc0 = pt_utils.Conv1d(3, 8, kernel_size=1, bn=True)

        self.dim = []

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.config.num_layers):
            d_out = self.config.d_out[i]
            # self.dilated_res_blocks.append(Dilated_res_block(d_in, d_out))
            self.dilated_res_blocks.append(BilateralAugmentation(d_in, d_out, self.config))
            d_in = 2 * d_out
            if i == 0:
                self.dim.append(d_in)
            self.dim.append(d_in)

        d_out = d_in
        self.decoder_0 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

        self.decoder_blocks = nn.ModuleList()
        self.decoder_blocks_2 = nn.ModuleList()
        self.weight_block = nn.ModuleList()
        self.decoder_idx = []
        start = 0
        for n in range(self.config.num_layers):
            self.decoder_idx.append(start)
            self.decoder_blocks.append(pt_utils.Conv2d(self.dim[-1 - n], self.dim[-1 - n], kernel_size=(1, 1), bn=True))
            d_in = self.dim[-1-n]
            for j in range(self.config.num_layers - n):
                start += 1
                self.decoder_blocks_2.append(pt_utils.Conv2d(self.dim[-j - 2 - n] + d_in,
                                                             self.dim[-j - 2 - n], kernel_size=(1, 1), bn=True))
                d_in = self.dim[-j - 2 - n]
            self.weight_block.append(
                pt_utils.Conv2d(self.dim[-j - 2 - n], 1, kernel_size=(1, 1), bn=True, activation=None))
        # print(self.decoder_blocks_2)
        self.fc1 = pt_utils.Conv2d(32, 64, kernel_size=(1, 1), bn=True)
        self.fc2 = pt_utils.Conv2d(64, 32, kernel_size=(1, 1), bn=True)
        self.dropout = nn.Dropout(0.5)
        self.fc3 = pt_utils.Conv2d(32, self.config.num_classes, kernel_size=(1, 1), bn=False, activation=None)
        if self.learn:
            self.ce_sigma = nn.Parameter(torch.Tensor([0.27]), requires_grad=True)
            self.l1_sigma = nn.Parameter(torch.Tensor([0.62]), requires_grad=True)
            # self.ce_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1.0), requires_grad=True)
            # self.l1_sigma = nn.Parameter(torch.Tensor(1).uniform_(0.2, 1.0), requires_grad=True)

    def forward(self, end_points):
        features = end_points['features']  # Batch*channel*npoints
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.config.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        # ###########################Decoder############################
        f_multi_decoder = []  # full-sized feature maps
        f_weights_decoders = []  # point-wise adaptive fusion weights
        for n in range(self.config.num_layers):
            feature = self.decoder_blocks[n](f_encoder_list[-1 - n])
            f_decoder_list = []
            for j in range(self.config.num_layers - n):
                f_interp_i = self.nearest_interpolation(feature, end_points['interp_idx'][-j - 1 - n])
                # print(torch.cat([f_encoder_list[-j - 2 - n], f_interp_i], dim=1).shape)
                f_decoder_i = self.decoder_blocks_2[self.decoder_idx[n] + j](
                    torch.cat([f_encoder_list[-j - 2 - n], f_interp_i], dim=1))
                feature = f_decoder_i
                f_decoder_list.append(f_decoder_i)
            f_multi_decoder.append(f_decoder_list[-1])
            # print(self.weight_block[n], f_decoder_list[-1].shape)
            curr_weight = self.weight_block[n](f_decoder_list[-1])
            f_weights_decoders.append(curr_weight)

        f_weights = torch.cat(f_weights_decoders, dim=1)
        f_weights = F.softmax(f_weights, dim=1)

        f_decoder_final = torch.zeros_like(f_multi_decoder[-1])
        for i in range(len(f_multi_decoder)):
            add = f_weights[:, i, :, :].unsqueeze(1).repeat(1, f_multi_decoder[i].size()[1], 1, 1)
            f_decoder_final = f_decoder_final + add * f_multi_decoder[i]

        features = self.fc1(f_decoder_final)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)

        if self.learn:
            return f_out, [self.ce_sigma, self.l1_sigma]
        else:
            return f_out

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.reshape(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        pool_features = pool_features.reshape(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class BilateralAugmentation(nn.Module):
    def __init__(self, d_in, d_out, config):
        """
        Bilateral Augmentation Block
        """
        super(BilateralAugmentation, self).__init__()
        self.config = config
        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp5 = pt_utils.Conv2d(d_out, 3, kernel_size=(1, 1), bn=True)
        self.mlp6 = pt_utils.Conv2d(9, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp7 = pt_utils.Conv2d(9, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp8 = pt_utils.Conv2d(d_out + d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.mlp9 = pt_utils.Conv2d(d_out, d_out, kernel_size=(1, 1), bn=False, activation=None)
        self.mlp10 = pt_utils.Conv2d(2 * d_out, d_out, kernel_size=(1, 1), bn=True)
        self.mlp11 = pt_utils.Conv2d(d_out, 2 * d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature, xyz, neigh_idx):
        feature = self.mlp1(feature)
        neigh_feat = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        neigh_feat = neigh_feat.permute((0, 3, 1, 2))
        neigh_xyz = self.gather_neighbour(xyz, neigh_idx)
        neigh_xyz = neigh_xyz.permute((0, 3, 1, 2))

        tile_feat = feature.repeat(1, 1, 1, self.config.k_n)
        tile_xyz = xyz.unsqueeze(2).repeat(1, 1, self.config.k_n, 1).permute((0, 3, 1, 2))

        feat_info = torch.cat([neigh_feat - tile_feat, tile_feat], dim=1)  # B, N, k, d_out
        neigh_xyz_offsets = self.mlp5(feat_info)

        shifted_neigh_xyz = neigh_xyz + neigh_xyz_offsets
        xyz_info = torch.cat([neigh_xyz - tile_xyz, shifted_neigh_xyz, tile_xyz], dim=1)  # B, N, k, d_out
        neigh_feat_offsets = self.mlp6(xyz_info)
        shifted_neigh_feat = neigh_feat + neigh_feat_offsets

        xyz_encoding = self.mlp7(xyz_info)
        feat_info = torch.cat([shifted_neigh_feat, feat_info], dim=1)
        feat_encoding = self.mlp8(feat_info)

        # Mixed Local Aggregation
        overall_info = torch.cat([xyz_encoding, feat_encoding], dim=1)
        k_weights = self.mlp9(overall_info)
        # print(k_weights.shape)
        'check'
        k_weights = F.softmax(k_weights, dim=-1)
        overall_info_weighted_sum = torch.sum(overall_info * k_weights, dim=-1, keepdim=True)  #

        'check'
        overall_info_max = overall_info.max(dim=-1, keepdims=True)[0]  # (B, N, 1, d)
        overall_encoding = torch.cat([overall_info_max, overall_info_weighted_sum], dim=1)

        overall_encoding = self.mlp10(overall_encoding)
        output_feat = self.mlp11(overall_encoding)  # B, N, 1, 2*d_out
        # return output_feat, shifted_neigh_xyz
        return output_feat

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features

class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        self.mlp1 = pt_utils.Conv2d(d_in, d_out // 2, kernel_size=(1, 1), bn=True)
        self.lfa = Building_block(d_out)
        self.mlp2 = pt_utils.Conv2d(d_out, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)
        self.shortcut = pt_utils.Conv2d(d_in, d_out * 2, kernel_size=(1, 1), bn=True, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(10, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_1 = Att_pooling(d_out, d_out // 2)

        self.mlp2 = pt_utils.Conv2d(d_out // 2, d_out // 2, kernel_size=(1, 1), bn=True)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2))  # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2))  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).repeat(1, 1, neigh_idx.shape[-1], 1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        # batch*npoint*nsamples*10
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.reshape(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
        features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    from dataset.semkitti_trainset import SemanticKITTI
    from dataset.poss_trainset import SemanticPOSS
    from torch.utils.data import DataLoader
    from config import ConfigSemanticKITTI as cfg

    train_dataset = SemanticPOSS('training')

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True,
        num_workers=0, collate_fn=train_dataset.collate_fn, )

    net = Network(cfg)

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of parameters: {} ".format(pytorch_total_params / 1000000) + "M")
    # exit()
    # print(net)
    net.cuda()
    for batch_idx, batch_data in enumerate(train_loader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(cfg.num_layers):
                    batch_data[key][i] = batch_data[key][i].cuda(non_blocking=True)
            else:
                batch_data[key] = batch_data[key].cuda(non_blocking=True)

        with torch.no_grad():
            end_points = net(batch_data)

        exit()