import torch
import numpy as np


# ------------------ IoU (Jaccard Index) ------------------
def compute_iou(pred, target, num_classes=10):
    """
    pred: [B, C, H, W] (logits)
    target: [B, H, W]
    """
    pred = torch.argmax(pred, dim=1)

    ious = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()

        if union == 0:
            continue
        else:
            ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return np.mean(ious)


# ------------------ Pixel Accuracy ------------------
def pixel_accuracy(pred, target):
    """
    pred: [B, C, H, W]
    target: [B, H, W]
    """
    pred = torch.argmax(pred, dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()

    return correct / total


# ------------------ Dice Score ------------------
def dice_score(pred, target, num_classes=10, smooth=1e-6):
    """
    Computes mean Dice score (not loss)
    """
    pred = torch.argmax(pred, dim=1)

    dice_scores = []

    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)

        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()

        if total == 0:
            continue

        dice = (2 * intersection + smooth) / (total + smooth)
        dice_scores.append(dice)

    if len(dice_scores) == 0:
        return 0.0

    return np.mean(dice_scores)