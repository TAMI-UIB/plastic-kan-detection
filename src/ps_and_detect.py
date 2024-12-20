import os

import hydra
import torch
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(os.environ["PROJECT_ROOT"], ".env"))

from callbacks.logger import TBoardLogger, ImagePlotCallback, GDriveLogger
from callbacks.convergence import WindowConvergence
from base import Experiment
from hydra.utils import instantiate


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    train_loader = instantiate(cfg.dataset.train)
    validation_loader = instantiate(cfg.dataset.validation)
    test_loader = instantiate(cfg.dataset.test)
    dataloaders = {"train": train_loader, "validation": validation_loader, "test": test_loader}

    experiment = Experiment(cfg)

    experiment.ps_model = instantiate(cfg.ps_model)
    ckpt = torch.load(f"/home/ivan/projects/plastic-kan-detection/logs/marida_w_index/x4/2024-10-07/ModelPansharpeningSharingOperators/default/version_15/checkpoints/best.ckpt")
    weights = ckpt['state_dict']
    weights = {".".join(k.split(".")[1:]): v for k, v in weights.items()}
    experiment.ps_model.load_state_dict(weights)

    tb_log_dir = f'{os.environ["LOG_DIR"]}/{cfg.dataset.name}/x{cfg.sampling}/{cfg.day}/{cfg.model.name}/'
    logger = TensorBoardLogger(tb_log_dir, name=cfg.nickname)
    csv_log_path = f"{os.environ['LOG_DIR']}/{cfg.dataset.name}"

    default_callbacks = [
        TBoardLogger(day=cfg.day, name=cfg.model.name),
        ImagePlotCallback(plot_interval=cfg.plot_interval),
        GDriveLogger(day=cfg.day, name=cfg.model.name, path=csv_log_path),
        RichModelSummary(max_depth=3),
        instantiate(cfg.checkpoint),
    ]

    callback_list = instantiate(cfg.model.callbacks) + default_callbacks if hasattr(cfg.model, 'callbacks') else default_callbacks
    trainer = Trainer(max_epochs=cfg.model.train.max_epochs, logger=logger,
                      devices=cfg.devices,
                      callbacks=callback_list, accelerator="auto")

    trainer.fit(experiment, train_loader, validation_loader)
    trainer.test(experiment, dataloaders=dataloaders)
    exit(0)

if __name__ == '__main__':
    train()
