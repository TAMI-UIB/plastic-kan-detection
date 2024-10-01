import numpy as np
import torch

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, \
    cohen_kappa_score, jaccard_score, accuracy_score

metrics_dict = {
    'accuracy': accuracy_score,
    'fscore': precision_recall_fscore_support,
    'auroc': roc_auc_score,
    'jaccard': jaccard_score,
    'kappa': cohen_kappa_score,
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
                if k == 'fscore':
                    _, _, f, _ = v(inputs[i].unsqueeze(0).cpu().detach().numpy().item(),
                                   targets[i].unsqueeze(0).cpu().detach().numpy().item(),
                                   zero_division=0, average="binary")
                    self.dict[k].append(f)
                else:
                    self.dict[k].append(v(inputs[i].unsqueeze(0).cpu().detach().numpy().item(),
                                          targets[i].unsqueeze(0).cpu().detach().numpy().item()))

    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}

    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}

    def get_dict(self):
        return self.dict
