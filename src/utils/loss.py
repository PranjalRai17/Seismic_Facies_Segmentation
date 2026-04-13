import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------ DICE LOSS ------------------
def dice_loss(pred, target, smooth=1e-6):
    """
    Multi-class Dice Loss
    pred: [B, C, H, W]
    target: [B, H, W]
    """
    pred = torch.softmax(pred, dim=1)
    target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2)

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


# ------------------ FOCAL LOSS ------------------
def focal_loss(pred, target, alpha=0.25, gamma=2):
    """
    Multi-class Focal Loss
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal = alpha * (1 - pt) ** gamma * ce_loss
    return focal.mean()


# ------------------ COMBINED LOSS ------------------
class CombinedLoss(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, target):
        f = focal_loss(pred, target)
        d = dice_loss(pred, target)
        return 0.5 * f + 0.5 * d