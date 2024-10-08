dependencies = ["torch", "pytorch_lightning", "segmentation_models_pytorch"]

from src.sota.marinedebrisdetector.marinedebrisdetector import SegmentationModel
from src.sota.marinedebrisdetector.marinedebrisdetector import CHECKPOINTS

__all__ = ['unetpp', 'unet']

def unetpp(seed=1, label_refinement=True):
    assert seed in [1,2,3]
    if label_refinement:
        return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}"], trust_repo=True)
    else:
        return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet++{seed}_no_label_refinement"], trust_repo=True)

def unet(seed=1):
    assert seed in [1,2,3]
    return SegmentationModel.load_from_checkpoint(CHECKPOINTS[f"unet{seed}"], trust_repo=True)

if __name__ == '__main__':
    unetpp(1)
