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

from callbacks.logger import MetricLogger, ImagePlotCallback, TestMetricLogger
from base import Experiment
from hydra.utils import instantiate


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def test(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))
    experiment = Experiment(cfg)
    validation_loader = instantiate(cfg.dataset.validation)
    test_loader = instantiate(cfg.dataset.test)

    for stage  in ["best", "last"]:
        tb_log_dir = f'{os.environ["LOG_DIR"]}/{cfg.dataset.name}/x{cfg.sampling}/{cfg.day}/{cfg.model.name}'
        logger = TensorBoardLogger(tb_log_dir, name=cfg.nickname)
        csv_log_path = f"{os.environ['LOG_DIR']}/{cfg.dataset.name}/{stage}"

        default_callbacks = [
                             MetricLogger(day=cfg.day, name=cfg.model.name, path_dir=csv_log_path),
                             ImagePlotCallback(plot_interval=cfg.plot_interval),
                             TestMetricLogger(day=cfg.day, name=cfg.model.name, path=csv_log_path),
                             RichModelSummary(max_depth=3),
                             instantiate(cfg.checkpoint),
                             ]

        callback_list = instantiate(cfg.model.callbacks) + default_callbacks if hasattr(cfg.model, 'callbacks') else default_callbacks
        trainer = Trainer(max_epochs=cfg.model.train.max_epochs, logger=logger, devices=cfg.devices,
                          callbacks=callback_list, accelerator="auto")

        ckpt = torch.load(f"{tb_log_dir}/default/version_{cfg.version}/checkpoints/{stage}.ckpt")
        weights = ckpt['state_dict']
        experiment.load_state_dict(weights)
        dataloaders = {"validation": validation_loader, "test": test_loader}

        trainer.test(experiment, dataloaders=dataloaders)
        exit(0)


if __name__ == '__main__':
    test()