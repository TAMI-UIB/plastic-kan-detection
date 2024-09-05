import pytorch_lightning as pl


class SaveImage(pl.Callback):
    def __init__(self, save_images):
        super().__init__()
        self.save_images = save_images

    def on_test_epoch_end(self, trainer, pl_module):
        if self.save_images:
            for subset, dataloader in trainer.test_dataloaders.items():
                for i, batch in enumerate(dataloader):
                    low, gt, name = batch
                    low = low.to(pl_module.device)
                    output = pl_module.forward(low)
                    pred_rgb = dataloader.dataset.get_rgb(output['pred'])
                    pl_module.save_image(low, gt['rgb'], name, pred_rgb, subset)