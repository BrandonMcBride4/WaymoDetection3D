from bbox.bbox3d import BBox3D
from bbox.metrics import iou_3d
from sklearn.metrics import auc
import torch
import numpy as np
import torch

def convertToBBox3D(torch_boxes):
    box3Ds = []
    for box in torch_boxes:
        box = box.numpy()
        box3Ds.append(BBox3D(box[0], box[1], box[2], length = box[3], width = box[4], height = box[5], is_center = True, euler_angles = [0, 0,box[6]]))
    return box3Ds

def pairwiseLabeledIOU(bboxes1, bboxes1_labels, bboxes2, bboxes2_labels):
    result = np.zeros((len(bboxes1), len(bboxes2)))
    for i, box1 in enumerate(bboxes1):
        for j, box2 in enumerate(bboxes2):
            result[i][j] = int(bboxes1_labels[i] == bboxes2_labels[j]) * iou_3d(box1, box2)
    return result

def computeMatches(gt_boxes, gt_labels, pred_boxes, pred_labels, threshold = 0.5):
    ious = pairwiseLabeledIOU(gt_boxes, gt_labels, pred_boxes, pred_labels)
    max = np.max(ious, axis = 0)
    return max > threshold

def computeAP(pred_labels, pred_scores, matches, class_id, num_bins = 100):
    thresholds = np.linspace(0, 1, num = num_bins)
    class_preds = pred_labels[pred_labels == class_id]
    class_matches = matches[pred_labels == class_id]
    class_scores = pred_scores[pred_labels == class_id]

    leveled_precisions = []
    for threshold in thresholds:
        pred_pos = np.sum(class_scores > threshold)
        pred_tp = np.sum(class_matches[class_scores > threshold])
        leveled_precisions.append(pred_tp/(pred_pos+1e-8))

    return auc(thresholds, leveled_precisions)

def computemAP(pred_labels, pred_scores, matches, num_classes = 3, num_bins = 100):
    APs = [computeAP(pred_labels, pred_scores, matches, class_id, num_bins = num_bins) for class_id in range(num_classes)]
    return np.mean(APs)