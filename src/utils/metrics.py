import numpy as np
import torch
from torchmetrics.functional.classification import binary_accuracy, dice, binary_jaccard_index, binary_precision, \
    binary_recall, binary_specificity

metrics_dict = {
    'precision': binary_precision,
    'recall': binary_recall,
    'specificity': binary_specificity,
    'accuracy': binary_accuracy,
    'dice': dice,
    'iou': binary_jaccard_index,
}
class MetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, inputs, targets):
        inputs = torch.where(torch.exp(inputs) > 0.5, 1, 0)
        for i in range(inputs.size(0)):
            for k, v in self.metrics.items():
                self.dict[k].append(v(inputs[i].unsqueeze(0), targets[i].unsqueeze(0), data_range=1).cpu().detach().numpy())


    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}


    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}
    def get_dict(self):
        return self.dict
