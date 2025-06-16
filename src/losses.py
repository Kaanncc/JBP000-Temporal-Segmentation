import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    
    probs = torch.sigmoid(logits)
    p = probs.view(-1)
    t = targets.view(-1)
    intersection = (p * t).sum()
    return 1 - (2 * intersection + eps) / (p.sum() + t.sum() + eps)


class TverskyLoss(nn.Module):
  
    def __init__(self, alpha: float = 0.9, beta: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        t = targets.view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        tversky_index = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky_index


class ComboLoss(nn.Module):
    
    def __init__(self, alpha: float = 0.9, bce_weight: float = 0.1):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha)
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = self.bce(logits, targets)
        loss_tversky = self.tversky(logits, targets)
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_tversky


class FocalTverskyLoss(nn.Module):
    
    def __init__(self, alpha: float = 0.9, beta: float = 0.1, gamma: float = 0.75, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p = probs.view(-1)
        t = targets.view(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return torch.pow((1 - ti), self.gamma)
