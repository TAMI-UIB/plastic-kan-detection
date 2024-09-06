import torch
import numpy as np


from torchmetrics.functional.image import structural_similarity_index_measure as SSIM
from torchmetrics.functional.image import spectral_angle_mapper as SAM
from torchmetrics.functional.image import error_relative_global_dimensionless_synthesis as ERGAS
from torchmetrics.functional.image import peak_signal_noise_ratio as PSNR

metrics_dict = {
    'ergas': ERGAS,
    'psnr': PSNR,
    'ssim': SSIM,
    'sam': SAM,
}
class MetricCalculator:
    def __init__(self,  metrics_list):
        self.metrics = {opt: metrics_dict[opt] for opt in metrics_list}
        self.dict = {k: [] for k in self.metrics.keys()}

    def update(self, ref, pred):
        for i in range(ref.size(0)):
            for k, v in self.metrics.items():
                match k:
                    case 'ergas' | 'sam':
                        self.dict[k].append(v(ref[i].unsqueeze(0), pred[i].unsqueeze(0)).cpu().detach().numpy())
                    case _:
                        self.dict[k].append(v(ref[i].unsqueeze(0), pred[i].unsqueeze(0), data_range=1).cpu().detach().numpy())


    def clean(self):
        self.dict = {k: [] for k in self.metrics.keys()}


    def get_statistics(self):
        mean_dict = {f"{k}": np.mean(v) for k, v in self.dict.items()}
        std_dict = {f"{k}": np.std(v) for k, v in self.dict.items()}
        return {"mean": mean_dict, "std": std_dict}
    def get_dict(self):
        return self.dict