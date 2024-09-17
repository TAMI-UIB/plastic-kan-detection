import numpy as np
import torch

from torchmetrics.functional.classification import binary_accuracy, dice, binary_precision, \
    binary_recall, binary_specificity


def IoU(inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # intersection is equivalent to True Positive count
        # union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        result = (intersection + smooth) / (union + smooth)

        return result


metrics_dict = {
    'precision': binary_precision,
    'recall': binary_recall,
    'specificity': binary_specificity,
    'accuracy': binary_accuracy,
    'dice': dice,
    'iou': IoU,
}

class MetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, inputs, targets):
        inputs = torch.where(torch.exp(inputs) > 0.5, 1, 0)
        targets = targets.int()
        for i in range(inputs.size(0)):
            for k, v in self.metrics.items():
                self.dict[k].append(v(inputs[i].unsqueeze(0), targets[i].unsqueeze(0)).cpu().detach().numpy())

    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}

    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}

    def get_dict(self):
        return self.dict
