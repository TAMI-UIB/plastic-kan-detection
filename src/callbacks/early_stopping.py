import math

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class WindowConvergence(Callback):
    def __init__(self, monitor="validation_loss", window_size=50, epsilon=0.01) -> None:
        super().__init__()
        self.monitor = monitor
        self.window_size = window_size
        self.epsilon = epsilon
        self.window = list()
        self.converged = False

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        trainer.should_stop = not self.converged

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        value = trainer.callback_metrics.get(self.monitor)
        self.window.append(value)
        if len(self.window) >= self.window_size:
            self.window = self.window[-self.window_size:]
            self.converged = abs(self._get_slope()) < self._lambda() * self.epsilon
        else:
            self.converged = False

    def _get_slope(self):
        x = np.array(list(range(self.window_size)))
        x = np.vstack([x, np.ones(len(x))]).T
        y = np.array(self.window)
        m, b = np.linalg.lstsq(x, y, rcond=None)[0]
        return m

    def _lambda(self):
        return 10 ** int(math.log10(np.mean(self.window)))
