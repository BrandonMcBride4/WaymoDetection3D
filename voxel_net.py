

#------------- 3D BACKBONE ----------------

from functools import partial
import torch
import torch.nn as nn 
import torch.nn.functional as F
import spconv
from datetime import datetime

def createSparseConvBlock(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=1,
                   conv_type='subm'):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )

    return m

class VoxelBackBone(spconv.SparseModule):
    def __init__(self, input_channels, grid_size):
        super().__init__()
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SparseConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            nn.BatchNorm1d(16, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        self.conv1 = createSparseConvBlock(16, 16, 3, indice_key='subm1')
    
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] -> [800, 704, 21]
            createSparseConvBlock(16, 32, 3, stride=2, indice_key='spconv2', conv_type='spconv'),
            createSparseConvBlock(32, 32, 3, indice_key='subm2'),
            createSparseConvBlock(32, 32, 3, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] -> [400, 352, 11]
            createSparseConvBlock(32, 64, 3, stride=2, indice_key='spconv3', conv_type='spconv'),
            createSparseConvBlock(64, 64, 3, indice_key='subm3'),
            createSparseConvBlock(64, 64, 3, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] -> [200, 176, 5]
            createSparseConvBlock(64, 64, 3, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            createSparseConvBlock(64, 64, 3, indice_key='subm4'),
            createSparseConvBlock(64, 64, 3, indice_key='subm4'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down2'),
            nn.BatchNorm1d(128, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

    def forward(self, batch_dict):
        voxel_features, voxel_coords = batch_dict['voxel_features'].contiguous(), batch_dict['voxel_coords'].contiguous()
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)

        return {
            'encoded_spconv_tensor': out,
            'multi_scale_3d_features': [x_conv1, x_conv2, x_conv3, x_conv4]
        }

#--------------- 2D BACKBONE -----------------------------

class ConvBlock(nn.Module):

    def __init__(self, in_channels, channels, kernel_size=3, padding=1, stride = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding, stride = stride, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class TConvBlock(nn.Module):

    def __init__(self, in_channels, channels, stride, padding):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, channels, stride, stride = stride, bias=False, padding = padding)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU()

    def forward(self, x, output_size = None):
        return self.relu(self.bn(self.conv(x, output_size = output_size)))


class BaseBEVBackbone(nn.Module):
    def __init__(self, input_channels):
        super().__init__()

        num_levels = 2
        layer_nums = [5, 5]
        layer_strides = [1, 2]
        num_filters = [128, 256]
        num_upsample_filters = [256, 256]
        upsample_strides = [1, 2]
        c_in_list = [input_channels, 128]
        
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            c_in, layer_num, layer_stride, num_filter, num_upsample_filter, upsample_stride = c_in_list[idx], layer_nums[idx], layer_strides[idx], num_filters[idx], num_upsample_filters[idx], upsample_strides[idx]
            level_layers = [ConvBlock(c_in, num_filter, stride = layer_stride)] + [ConvBlock(num_filter, num_filter) for _ in range(layer_num)]
            self.blocks.append(nn.Sequential(*level_layers))
            sparse_pad = 0 if layer_stride == 1 else 1
            self.deblocks.append(TConvBlock(num_filter, num_upsample_filter, upsample_stride, sparse_pad))
        self.num_upsample_filters = num_upsample_filters

    def forward(self, spatial_features):
        ups = []
        x = spatial_features
        size = x.shape
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x, output_size = (size[0], self.num_upsample_filters[i], size[2], size[3])))
        return torch.cat(ups, dim=1)


#-------------------- DENSE HEADS ------------------------------------

class DenseHead(nn.Module):
    def __init__(self, input_channels, num_class, num_anchors_per_location, box_size):
        super().__init__()
 
        self.num_class = num_class
        self.num_anchors_per_location = num_anchors_per_location
        self.box_size = box_size
        self.num_dir_bins = 2

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_size,
            kernel_size=1
        )
        self.conv_dir_cls = nn.Conv2d(
            input_channels,
            self.num_anchors_per_location * self.num_dir_bins,
            kernel_size=1
        )

        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, spatial_features):
        cls_preds = self.conv_cls(spatial_features)
        box_preds = self.conv_box(spatial_features)
        dir_cls_preds = self.conv_dir_cls(spatial_features)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()

        return {'cls_preds'     : cls_preds, 
                'box_preds'     : box_preds, 
                'dir_cls_preds' : dir_cls_preds}


#-------------------- BEV CONVERSION -----------------------------------

def to_BEV(batch_features):
    encoded_spconv_tensor = batch_features['encoded_spconv_tensor']
    spatial_features = encoded_spconv_tensor.dense()
    N, C, D, H, W = spatial_features.shape
    return spatial_features.view(N, C * D, H, W)

#---------------------- LOSSES ---------------------------------


class SigmoidFocalClassificationLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def sigmoid_cross_entropy_with_logits(self, input, target):
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.
        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super(WeightedSmoothL1Loss, self).__init__()
        self.code_weights = np.array(code_weights, dtype=np.float32)
        self.code_weights = torch.from_numpy(self.code_weights)#.cuda()
        self.beta = beta
    def smooth_l1_loss(self, diff, beta):
        n = torch.abs(diff)
        loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.
        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # # code-wise weighting
        # if self.code_weights is not None:
        #     diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.
        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss

class RPNLoss():

    def __init__(self, num_class, num_dir_bins):
        self.num_class = num_class
        self.num_dir_bins = num_dir_bins
        self.lambda_cls = 1.0
        self.lambda_box = 2.0
        self.lambda_dir = 1.0
        self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.dir_loss = WeightedCrossEntropyLoss()
        self.reg_loss = WeightedSmoothL1Loss(code_weights = self.code_weights)
        self.cls_loss = SigmoidFocalClassificationLoss()

    def compute_loss(self, predictions, ground_truth):
        batch_size = ground_truth['gt_class'].shape[0]
        box_cls_gt =  ground_truth['gt_class'].permute(0,2,3,1).reshape(batch_size, -1, self.num_class).contiguous()
        box_reg_gt = ground_truth['gt_reg'].permute(0,2,3,1).reshape(batch_size, -1, ground_truth['gt_reg'].shape[1]).contiguous()
        box_dir_gt = ground_truth['gt_dir'].permute(0,2,3,1).reshape(batch_size, -1, ground_truth['gt_dir'].shape[1]).contiguous().float()
        cls_loss = self.compute_cls_loss(predictions['cls_preds'],box_cls_gt)
        box_loss, dir_loss = self.compute_box_dir_loss(predictions['box_preds'], predictions['dir_cls_preds'],
                                                       box_reg_gt, box_dir_gt, box_cls_gt)
        total_loss = cls_loss * self.lambda_cls + box_loss * self.lambda_box + dir_loss * self.lambda_dir
        loss_dict = {'cls_loss': cls_loss.item(), 'box_loss': box_loss.item(), 'dir_loss': dir_loss.item(), 'total_loss': total_loss.item()}
        return total_loss, loss_dict

    def compute_cls_loss(self, cls_preds, box_cls_labels):

        batch_size = int(cls_preds.shape[0])
        negatives = (box_cls_labels[:,:,0] != 0).float()  # [N, num_anchors]
        positives = (box_cls_labels[:,:,0] == 0).float()

        pos_normalizer = negatives.mean(1, keepdim=True).float()
        neg_normalizer = 1 - pos_normalizer
        cls_weights = positives * pos_normalizer + negatives * neg_normalizer

        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        cls_loss_src = self.cls_loss(cls_preds, box_cls_labels, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        return cls_loss

    def compute_box_dir_loss(self, box_preds, box_dir_cls_preds, box_reg_targets, box_dir_targets, box_cls_labels):

        batch_size = int(box_preds.shape[0])
        reg_weights = (box_cls_labels[:,:,0] == 0).float()
        pos_normalizer = reg_weights.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        box_preds = box_preds.view(batch_size, -1, box_preds.shape[-1])# // self.num_anchors_per_location)

        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        box_loss = loc_loss_src.sum() / batch_size

        dir_logits = box_dir_cls_preds.view(batch_size, -1, self.num_dir_bins)
        dir_loss = self.dir_loss(dir_logits, box_dir_targets, weights=reg_weights)
        dir_loss = dir_loss.sum() / batch_size

        return box_loss, dir_loss

    def add_sin_difference(self, boxes1, boxes2, dim=6):
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

#------------------------------ LIGHTENING MODULE ----------------------------------

import pytorch_lightning as pl
import torch.optim as optim
import numpy as np

class lightningVoxelNet(pl.LightningModule):
    def __init__(self, save_dir):
        super().__init__()
        input_channels = 4
        grid_size = np.array([1540, 1540, 40])
        box_size = 7
        bev_input_channels = 256
        dense_head_input_channels = 512
        num_class = 4
        num_anchors_per_location = 1
        num_dir_bins = 2
        
        self.save_dir = save_dir
        self.backbone3d = VoxelBackBone(input_channels, grid_size)
        self.to_BEV = to_BEV
        self.backbone2d = BaseBEVBackbone(bev_input_channels)
        self.densehead = DenseHead(dense_head_input_channels, num_class, num_anchors_per_location, box_size)
        self.loss_module = RPNLoss(num_class, num_dir_bins)

    def forward(self, batch_dict):
        voxel_features = self.backbone3d(batch_dict)
        bev_features = self.to_BEV(voxel_features)
        bev_features = self.backbone2d(bev_features)
        predictions = self.densehead(bev_features)
        return predictions
    
    def training_step(self, batch_dict, batch_idx):
        prediction = self(batch_dict)
        loss, loss_dict = self.loss_module.compute_loss(prediction, batch_dict)
        self.logger.log_metrics(loss_dict)
        return loss

    def training_epoch_end(self, epoch_results):
        self.save_model()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-4, weight_decay = 0, eps = 1e-5)
        #scheduler = optim.lr_scheduler.CyclicLR(optimizer, 5e-4, 1e-2, base_momentum = 0.7, step_size_up=15, step_size_down = 30, cycle_momentum = False)
        return optimizer

    def save_model(self):
        now = datetime.now()
        dt_string = now.strftime("%d.%H.%M")
        fname = "model" + dt_string + '.pt'
        save_path = self.save_dir + fname
        print(f"Saving Model To: {save_path}")
        torch.save(self.state_dict(), save_path)


