import numpy as np
import torch
import torchvision.ops as ops


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate
    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes

def get_boxes_global(cls, reg, dir, class_idx=-1, pred_thresh=0.5):
    """

    :param cls: [4, 193, 193]
    :param reg: [7, 193, 193]
    :param dir: [2, 193, 193]
    :param pred_thresh: scalar
    :return: [N, 7] (x, y, z, dx, dy, dz, heading)
    """
    world_space_offset = torch.tensor([-75.2, -75.2, -2]).unsqueeze(1)
    output_voxel_size = torch.tensor([75.2 * 2 / 193, 75.2 * 2 / 193, 6]).unsqueeze(1)

    max_value, max_class = torch.max(cls, dim=0)
    if class_idx == -1:
        mask = (max_class > 0) & (max_value > pred_thresh)
    else:
        mask = (max_class == class_idx) & (max_value > pred_thresh)

    x, y = torch.meshgrid(torch.arange(193), torch.arange(193))
    x = x[mask]
    y = y[mask]

    box = reg[:, mask]

    box[:3] = box[:3] + (torch.vstack([x.unsqueeze(0), y.unsqueeze(0), torch.zeros(x.shape).unsqueeze(0)]) + .5)\
              * output_voxel_size + world_space_offset
    box[6] = box[6] * (torch.argmax(dir[:, mask].to(torch.int), dim=0) - 0.5) * 2
    return box.T


def nms(pred, pred_thresh=0.5, iou_thresh=0.5, num_class=4):
    """

    :param pred:
    :param pred_thresh:
    :param iou_thresh:
    :param num_class:
    :return: list of batch size of lists of number of classes of 3d world boxes (N, 7)(x, y, z, dx, dy, dz, yaw)
    """

    batch_size = pred['box_preds'].shape[0]
    nms_list = []
    for i in range(batch_size):
        cls = pred['cls_preds'][i].permute(2, 0, 1)
        reg = pred['box_preds'][i].permute(2, 0, 1)
        dir = pred['dir_cls_preds'][i].permute(2, 0, 1)
        cls_value, max_class = torch.max(cls, dim=0)
        nms_obj_idxs = []
        for c in range(1, num_class):
            mask = (max_class == c) & (cls_value > pred_thresh)
            mask_idx = mask.nonzero(as_tuple=False)
            if mask_idx.shape[0] > 0:
                scores = cls[c, mask]
                boxes3d = get_boxes_global(cls, reg, dir, class_idx=c, pred_thresh=pred_thresh)
                boxes = boxes3d_lidar_to_aligned_bev_boxes(boxes3d)
                nms_idx = ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_thresh)
                nms_obj_idxs.append(boxes3d[nms_idx])
            else:
                nms_obj_idxs.append([])
        nms_list.append(nms_obj_idxs)

    return nms_list

