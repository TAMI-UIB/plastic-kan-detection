import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image

from .utils import s2_to_rgb, calculate_fdi, calculate_ndvi
from .upload_drive import download_drive, upload_drive


class TestMetricLogger(Callback):
    def __init__(self, day, name, path) -> None:
        super(TestMetricLogger, self).__init__()
        self.name = name
        self.path = path
        self.day = day

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for subset in trainer.test_dataloaders.keys():
            os.makedirs(f'{self.path}/reports/', exist_ok=True)
            file_name = f'{self.path}/reports/{subset}.csv'
            try:
                csv_logger = pd.read_csv(file_name)
            except FileNotFoundError:
                csv_logger = pd.DataFrame()
            metrics = pl_module.metrics[subset].get_statistics()
            data = {"day": [str(self.day)], "model": [self.name],
                    **{key: [value] for key, value in metrics['mean'].items()}}
            new_data = pd.DataFrame(data)
            csv_logger = pd.concat([csv_logger, new_data])
            csv_logger.to_csv(file_name, index=False)


class TBoardLogger(Callback):
    def __init__(self, day, name) -> None:
        super(TBoardLogger, self).__init__()
        fit_subsets = ['train', 'validation']
        self.best_metrics = {k: {} for k in fit_subsets}
        self.day = day
        self.name = name
        self.best_metric = -99999

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for subset in pl_module.fit_subsets:
            pl_module.metrics[subset].clean()
            pl_module.loss[subset] = 0
            for k, v in pl_module.loss_components[subset].items():
                pl_module.loss_components[subset][k] = 0

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        logger = trainer.logger
        writer = logger.experiment
        epoch = trainer.current_epoch
        writer.add_scalars(f"loss/comparison", {k: v for k, v in pl_module.loss.items() }, epoch)
        all_statistics = {k: v.get_statistics() for k, v in pl_module.metrics.items() if k in pl_module.fit_subsets}
        # Log the reference metric for saving checkpoints
        stage = pl_module.cfg.checkpoint.monitor.split('_')[0]
        metric = pl_module.cfg.checkpoint.monitor.split('_')[-1]
        pl_module.log(pl_module.cfg.checkpoint.monitor, all_statistics[stage]['mean'][metric], prog_bar=True)
        for metric in all_statistics['train']['mean'].keys():
            writer.add_scalars(f"{metric}/comparison", {k: v['mean'][metric] for k, v in all_statistics.items()}, epoch)
        for subset in pl_module.fit_subsets:
            statistics = all_statistics[subset]
            for k, v in statistics['mean'].items():
                writer.add_scalar(f"{k}/{subset}", v, epoch)
            writer.add_scalars(f"loss/{subset}_components", pl_module.loss_components[subset], epoch)
            writer.add_scalar(f"loss/{subset}", pl_module.loss[subset], epoch)
            if subset == "validation" and statistics['mean'][metric] > self.best_metric: # TODO: Reformulate this to do it dynamical based on pl_module.cfg.checkpoint.mode
                self.best_metric = statistics['mean'][metric]
                self.best_metrics['validation'] = statistics['mean']
                writer.add_text("best_metrics", str(statistics['mean']), global_step=epoch)


class GDriveLogger(Callback):
    def __init__(self, day, name, path) -> None:
        super().__init__()
        self.day = day
        self.name = name
        self.path = path

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for subset in pl_module.subsets:
            pl_module.metrics[subset].clean()

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        subsets = pl_module.subsets
        for subset in subsets:
            metrics = pl_module.metrics[subset].get_statistics()
            os.makedirs(f'{self.path}/reports/', exist_ok=True)
            file_name = f'{self.path}/reports/{subset}.csv'
            download_drive(self.path, subset, pl_module.cfg.dataset.name)
            csv_logger = pd.read_csv(file_name) if os.path.exists(file_name) else pd.DataFrame()
            data = {"day": [str(self.day)],
                    "model": [self.name],
                    "nickname": [pl_module.cfg.nickname],
                    "parameters": [pl_module.num_params],
                    **{key: [value] for key, value in metrics['mean'].items()}, "log_path": [trainer.log_dir]}
            print(data)
            new_data = pd.DataFrame(data)
            csv_logger = pd.concat([csv_logger, new_data])
            csv_logger.to_csv(file_name, index=False)
            if os.environ['UPLOAD_FILES'] == 'True':
                upload_drive(file_name, pl_module.cfg.dataset.name)




class ImagePlotCallback(pl.Callback):
    def __init__(self, plot_interval=200):
        super().__init__()
        self.plot_interval = plot_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        # Verificar si la época actual es múltiplo de plot_interval
        if trainer.current_epoch % self.plot_interval == 0:

            # Establece el modo de evaluación
            pl_module.eval()
            with torch.no_grad():
                # Recorre el DataLoader de validación
                batch = next(iter(trainer.val_dataloaders))
                images, masks, name = batch
                images = images.to(pl_module.device)

                preds = pl_module(images)

            # Vuelve al modo de entrenamiento
            pl_module.train()

            N = min(max(masks.shape[0],2), 5)

            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            preds = preds.cpu().numpy()

            # Crea un grid de imatges
            preds_exp = np.where(np.exp(preds) > 0.5, 1, 0)
            preds_sig = np.where(torch.sigmoid(torch.tensor(preds)).detach().cpu().numpy() > 0.5, 1, 0)

            height = 3
            width = 3
            images = np.transpose(images, (0, 2, 3, 1))
            masks = np.transpose(masks, (0, 2, 3, 1))
            preds = np.transpose(preds, (0, 2, 3, 1))
            preds_exp = np.transpose(preds_exp, (0, 2, 3, 1))
            preds_sig = np.transpose(preds_sig, (0, 2, 3, 1))
            fig, axs = plt.subplots(N, 6, figsize=(6 * width, N * height), squeeze=False)
            for axs_row, img, mask, pred, pred_exp, pred_sig in zip(axs, images, masks, preds, preds_exp, preds_sig):
                axs_row[0].imshow(s2_to_rgb(img))
                axs_row[0].set_title("RGB")
                axs_row[1].imshow(calculate_ndvi(img), cmap="viridis")
                axs_row[1].set_title("NDVI")
                axs_row[2].imshow(calculate_fdi(img), cmap="magma")
                axs_row[2].set_title("FDI")
                axs_row[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
                axs_row[3].set_title("Mask")
                axs_row[4].imshow(pred, cmap='gray', vmin=-1, vmax=2)
                axs_row[4].set_title("Prediction")
                axs_row[5].imshow(pred_exp, cmap='gray', vmin=0, vmax=1)
                axs_row[5].set_title("Binary prediction")

                [ax.axis("off") for ax in axs_row]

            plt.tight_layout()
            # Agregar la imagen a TensorBoard
            trainer.logger.experiment.add_figure(f'{pl_module.cfg.model.name}_{pl_module.cfg.nickname}/gt_vs_pred', fig,
                                                 trainer.current_epoch)


class SaveImageCallback(pl.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_test_epoch_end(self, trainer, pl_module):
        for subset, dataloader in trainer.test_dataloaders.items():
            for batch in dataloader:
                images, masks, name = batch
                images = images.to(pl_module.device)
                preds = pl_module(images)
                images = images.cpu().numpy()
                i = 0
                save_image(torch.permute(torch.from_numpy(s2_to_rgb(images[i])), (2, 0, 1)), f"{self.path}/{i}_RGB.png")
                plt.imshow(calculate_ndvi(images[i]), cmap="viridis")
                plt.axis("off")
                plt.savefig(f"{self.path}/{i}_NDVI.png")
                plt.imshow(calculate_fdi(images[i]), cmap="magma")
                plt.axis("off")
                plt.savefig(f"{self.path}/{i}_FDI.png")
                save_image(masks[0][i], f"{self.path}/{i}_GT.png")
                save_image(preds[0][i], f"{self.path}/{i}_pred.png")


class TestMetricPerImage(Callback):
    def __init__(self, ) -> None:
        super(TestMetricPerImage, self).__init__()

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        nickname = pl_module.cfg.nickname
        dataset = pl_module.cfg.dataset.name
        filename = [f'{os.environ["LOG_DIR"]}/{dataset}/evaluation-report/{nickname}_{k}_sampling_{pl_module.cfg.sampling}.csv' for k in trainer.test_dataloaders.keys()]
        subset = trainer.test_dataloaders.keys()
        for i, subset in enumerate(subset):
            metric_dict = pl_module.metrics[subset].get_dict()
            csv_logger = pd.DataFrame(metric_dict)
            csv_logger.to_csv(filename[i], index=True)

# if __name__ == '__main__':
#     from src.dataset.sentinel2 import sentinel2
#     from dotenv import load_dotenv
#     load_dotenv()
#
#     dataval = sentinel2(root=f"{os.environ['DATASET_PATH']}/sentinel2", fold="validation", output_size=64)
#     dataloader = torch.utils.data.DataLoader(dataset=dataval, batch_size=1, shuffle=False, num_workers=2)
#     model = UNet( channels= 12, hidden_channels=64)
#
#      # Recorre el DataLoader de validación
#     batch = next(iter(dataloader))
#     images, masks, name = batch
#
#     preds = model(images)
#
#     N = min(max(masks.shape[0], 2), 5)
#
#     images = images.detach().numpy()
#     masks = masks.detach().numpy()
#     preds = preds.detach().numpy()
#
#     # Crea un grid de imatges
#     preds_exp = np.where(np.exp(preds) > 0.5, 1, 0)
#     preds_sig = np.where(torch.sigmoid(torch.tensor(preds)).detach().cpu().numpy() > 0.5, 1, 0)
#
#     height = 3
#     width = 3
#     print(images.shape)
#     images = np.transpose(images, (0, 2, 3, 1))
#     masks = np.transpose(masks, (0, 2, 3, 1))
#     preds = np.transpose(preds, (0, 2, 3, 1))
#     preds_exp = np.transpose(preds_exp, (0, 2, 3, 1))
#     preds_sig = np.transpose(preds_sig, (0, 2, 3, 1))
#     print(f'images shape: {images.shape}')
#     print(f'masks shape: {masks.shape}')
#     print(f'preds shape: {preds.shape}')
#     print(f'preds_exp shape: {preds_exp.shape}')
#     print(f'preds_sig shape: {preds_sig.shape}')
#     fig, axs = plt.subplots(N, 6, figsize=(6 * width, N * height), squeeze=False)
#     for axs_row, img, mask, pred, pred_exp, pred_sig in zip(axs, images, masks, preds, preds_exp, preds_sig):
#
#         axs_row[0].imshow(s2_to_rgb(img))
#         axs_row[0].set_title("RGB")
#         axs_row[1].imshow(calculate_ndvi(img), cmap="viridis")
#         axs_row[1].set_title("NDVI")
#
#         axs_row[2].imshow(calculate_fdi(img), cmap="magma")
#         axs_row[2].set_title("FDI")

#         axs_row[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
#         axs_row[3].set_title("Mask")
#         axs_row[4].imshow(pred, cmap='gray', vmin=-1, vmax=2)
#         axs_row[4].set_title("Prediction")
#         axs_row[5].imshow(pred_exp, cmap='gray', vmin=0, vmax=1)
#         axs_row[5].set_title("Binary prediction")
#
#         [ax.axis("off") for ax in axs_row]
#
#     plt.tight_layout()
#     plt.savefig("../tmp/fig.png")


