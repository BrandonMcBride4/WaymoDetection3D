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


def nms(pred, iou_thresh=0.5, box_size=7, num_class=4):
    batch_size = pred['box_preds'].shape[0]
    nms_list = []
    for i in range(batch_size):
        cls = pred['cls_preds'][i].view(-1, num_class)
        cls_idx = torch.argmax(cls, dim=1)
        nms_obj_idxs = []
        for c in range(1, num_class):
            obj_idx = (cls_idx == c).nonzero(as_tuple=False)
            scores = cls[obj_idx[:, 0], c]
            boxes3d = pred['box_preds'][i].view(-1, box_size)[obj_idx[:, 0]]
            boxes = boxes3d_lidar_to_aligned_bev_boxes(boxes3d)
            nms_idx = ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_thresh)
            nms_obj_idxs.append(obj_idx[nms_idx, 0])
        nms_list.append(torch.cat(nms_obj_idxs))

    return nms_list

