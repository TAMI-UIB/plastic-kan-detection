import numpy as np
import torch

from torchmetrics.functional.classification import (binary_accuracy, binary_f1_score, binary_auroc, binary_jaccard_index,
                                                    binary_cohen_kappa)

def jaccard(pred, target):
    pred = torch.where(pred > 0.5, 1., 0.)
    target = target.int()

    inter = target * pred
    union = target + pred - inter
    return inter.sum() / union.sum()


metrics_dict = {
    'accuracy': binary_accuracy,
    'fscore': binary_f1_score,
    'auroc': binary_auroc,
    'jaccard': jaccard,
    'kappa': binary_cohen_kappa,
}

class MetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.int()
        if torch.sum(targets) == 0 or torch.sum(preds) == 0:
            pass
        else:
            for i in range(preds.size(0)):
                for k, v in self.metrics.items():
                    if k == "jaccard":
                        value = v(preds[i].unsqueeze(0), targets[i].unsqueeze(0))
                        if torch.isnan(value).any():
                            preds_aux = torch.where(preds > 0.5, 1., 0.)
                            jaccard = torch.sum(targets[i] * preds_aux) / torch.sum(
                                torch.where(targets[i] + preds_aux == 2, 1, targets[i] + preds_aux))
                            print(jaccard)
                            self.dict[k].append(jaccard)

                    self.dict[k].append(v(preds[i].unsqueeze(0), targets[i].unsqueeze(0)).cpu().detach().numpy())

    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}

    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v).item() for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v).item() for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}

    def get_dict(self):
        return self.dict


if __name__ == '__main__':
    metrics = MetricCalculator(['accuracy', 'fscore', 'auroc', 'jaccard', 'kappa'])

    preds = torch.rand((1,1024,1024))
    targets = torch.rand((1,1024,1024))
    targets = torch.where(targets > 0.5, 1, 0)

    metrics.update(preds, targets)
    print(metrics.get_statistics())