import torch.nn as nn
import torch
import torch.nn.functional as F
from functorch.einops import rearrange
from torchmetrics.functional import spectral_angle_mapper

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs,  targets)
        loss = (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def components(self):
        return []

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (torch.sum(inputs) + torch.sum(targets) + smooth)

        return 1 - dice, {'dice': 1 - dice}


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def components(self):
        return ['dice', 'bce']

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)
        inputs = rearrange(inputs, 'b c h w -> b (h w c)')
        targets = rearrange(targets, 'b c h w -> b (h w c)')

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        Dice_BCE = 0.5 * BCE + 0.5 * torch.mean(dice_loss)

        return Dice_BCE, {'dice': dice_loss, 'bce': BCE}

class DiceBCEIOULoss(nn.Module):
    def __init__(self):
        super(DiceBCEIOULoss, self).__init__()

    def components(self):
        return ['bce', 'dice', 'iou']

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        iou = self._iou(inputs, targets)
        Dice_BCE_IOU = BCE + dice_loss + iou

        return Dice_BCE_IOU, {'bce': BCE, 'dice': dice_loss, 'iou': iou}

    def _iou(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def components(self):
        return []

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU, {'IoU': 1-IoU}

class PSLoss(nn.Module):
    def __init__(self):
        super(PSLoss, self).__init__()

    def components(self):
        return ['mse']

    def forward(self, pred, target):
        mse = self.MSEloss(pred, target)
        return mse, {'mse': mse}
