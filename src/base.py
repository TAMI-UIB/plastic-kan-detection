import os
from typing import Dict, Any

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchvision.utils import save_image

pl.seed_everything(42)
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

    def forward(self, low):
        return self.model(low)

    def training_step(self, input, idx):
        low, gt, name = input
        output = self.forward(low)
        loss, loss_dict = self.loss_criterion(output, gt)
        self.metrics['train'].update(preds=output, targets=gt)
        self.loss_report(loss, loss_dict, 'train')
        return loss

    def validation_step(self, input, idx):
        low, gt, name = input
        output = self.forward(low)
        loss, loss_dict = self.loss_criterion(output, gt)
        self.metrics['validation'].update(preds=output, targets=gt)
        self.loss_report(loss, loss_dict, 'validation')
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        low, gt, name = batch
        print(dataloader_idx)
        output = self.forward(low)
        self.metrics['test'].update(preds=output, targets=gt)

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
