import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import directed_hausdorff


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
    
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6) -> torch.Tensor:
   
    pred = pred.float().contiguous().view(-1)
    target = target.float().contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def accuracy_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    
    pred = pred.int().view(-1)
    target = target.int().view(-1)
    correct = (pred == target).sum().float()
    return correct / pred.numel()


def precision_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
   
    pred = pred.int().view(-1)
    target = target.int().view(-1)
    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    return tp / (tp + fp + eps)


def recall_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
   
    pred = pred.int().view(-1)
    target = target.int().view(-1)
    tp = ((pred == 1) & (target == 1)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    return tp / (tp + fn + eps)


def temporal_stability(pred_seq: torch.Tensor, eps=1e-6) -> torch.Tensor:  ## MIF-IOU RENAME
    
    if pred_seq.dim() == 4:
        B, T, H, W = pred_seq.shape
        pred_seq = pred_seq.view(B * T, H, W)
        T = pred_seq.shape[0]
    else:
        T = pred_seq.shape[0]
    if T < 2:
        return torch.tensor(float('nan'), device=pred_seq.device)

    values = []
    for k in range(T - 1):
        inter = (pred_seq[k].float() * pred_seq[k + 1].float()).sum()
        union = (pred_seq[k].float().sum() + pred_seq[k + 1].float().sum() - inter + eps)
        values.append(inter / union)

    return torch.stack(values).mean()



def stability_rate(pred_seq: torch.Tensor, eps=1e-6) -> torch.Tensor:
    
    if pred_seq.dim() == 4:
        B, T, H, W = pred_seq.shape
        pred_seq = pred_seq.view(B * T, H, W)
        T = pred_seq.shape[0]
    else:
        T = pred_seq.shape[0]
    if T < 3:
        return torch.tensor(float('nan'), device=pred_seq.device)

    prev_iou = None
    deltas = []
    for k in range(T - 1):
        inter = (pred_seq[k].float() * pred_seq[k + 1].float()).sum()
        union = (pred_seq[k].float().sum() + pred_seq[k + 1].float().sum() - inter + eps)
        curr_iou = inter / union
        if prev_iou is not None:
            deltas.append(torch.abs(curr_iou - prev_iou))
        prev_iou = curr_iou

    return 1 - torch.stack(deltas).mean()
