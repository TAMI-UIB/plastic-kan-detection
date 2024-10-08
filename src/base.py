import os
from typing import Dict, Any

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.utils import save_image

from utils.images import calculate_fdi, calculate_ndvi

pl.seed_everything(42)

bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
indexes20m = [bands.index(band) for band in ['B5', 'B6', 'B7', 'B8A']]


class Experiment(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(Experiment, self).__init__()
        # Experiment configuration
        self.cfg = cfg
        # Define subsets
        self.subsets = ['train', 'validation', 'test']
        self.fit_subsets = ['train', 'validation']
        # Define model and loss
        self.model = instantiate(cfg.model.module)
        self.loss_criterion = instantiate(cfg.model.loss)
        # Number of model parameters
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Metric calculator
        self.metrics = {k: instantiate(cfg.metrics) for k in self.subsets}
        # Loss report
        self.loss = {subset: 0 for subset in self.fit_subsets}
        self.loss_components = {subset: {k: 0 for k in self.loss_criterion.components()} for subset in self.fit_subsets}
        self.ps_model = None

    def forward(self, **kwargs):
        if self.ps_model:
            images = kwargs.pop('image')
            highres = self.ps_model(**kwargs)
            print(images.shape, highres.shape)
            images[:, indexes20m, :, :] = highres
            images = images[:, 0:12, :, :]
            fdis = torch.Tensor(np.array([[calculate_fdi(image)] for image in images.cpu().detach().numpy()]))
            ndvis = torch.Tensor(np.array([[calculate_ndvi(image)] for image in images.cpu().detach().numpy()]))
            kwargs.update({'x': torch.cat([images, fdis, ndvis], dim=1)})
        return self.model(**kwargs)

    def training_step(self, input, idx):
        gt = input.pop('gt')
        output = self.forward(**input)
        loss, loss_dict = self.loss_criterion(pred=output, target=gt)
        self.metrics['train'].update(preds=output, targets=gt)
        self.loss_report(loss, loss_dict, 'train')
        return loss

    def validation_step(self, input, idx):
        gt = input.pop('gt')
        output = self.forward(**input)
        loss, loss_dict = self.loss_criterion(pred=output, target=gt)
        self.metrics['validation'].update(preds=output, targets=gt)
        self.loss_report(loss, loss_dict, 'validation')
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        gt = batch['gt']
        output = self.forward(**batch)
        self.metrics[self.subsets[dataloader_idx]].update(preds=output, targets=gt)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.model.optimizer,params=self.parameters())
        scheduler = instantiate(self.cfg.model.scheduler, optimizer=optimizer)
        return {'optimizer': optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['cfg'] = self.cfg
        checkpoint['current_epoch'] = self.current_epoch

    def loss_report(self, loss, loss_dict, subset):
        self.loss[subset] += loss
        for subset in self.fit_subsets:
            for k, v in self.loss_components[subset].items():
                self.loss_components[subset][k] += loss_dict[k]

    def save_image(self, low, gt, name, pred, subset):
        img_path = f'{self.logger.log_dir}/images/{subset}/'
        os.makedirs(img_path, exist_ok=True)
        for i in range(low.size(0)):
            save_image(low[i], f'{img_path}{name[i].split("_")[-1]}_low.png')
            save_image(gt[i], f'{img_path}{name[i].split("_")[-1]}_gt.png')
            save_image(pred[i], f'{img_path}{name[i].split("_")[-1]}_{self.cfg.model.name}.png')
