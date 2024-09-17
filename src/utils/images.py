import numpy as np
import rasterio as rio
from skimage.exposure import equalize_hist

L1CBANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
L2ABANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]


def pad(image, mask=None, output_size=64):
    # if feature is near the image border, image wont be the desired output size
    H, W = output_size, output_size
    c, h, w = image.shape
    dh = (H - h) / 2
    dw = (W - w) / 2
    image = np.pad(image, [(0, 0), (int(np.ceil(dh)), int(np.floor(dh))),
                           (int(np.ceil(dw)), int(np.floor(dw)))])

    if mask is not None:
        mask = np.pad(mask, [(int(np.ceil(dh)), int(np.floor(dh))),
                             (int(np.ceil(dw)), int(np.floor(dw)))])

        return image, mask
    else:
        return image


def read_tif_image(imagefile, window=None):
    # loading of the image
    with rio.open(imagefile, "r") as src:
        image = src.read(window=window)

        is_l1cimage = src.meta["count"] == 13  # flag if l1c (top-of-atm) or l2a (bottom of atmosphere) image

        # keep only 12 bands: delete 10th band (nb: 9 because start idx=0)
        if is_l1cimage:  # is L1C Sentinel 2 dataset
            image = image[[L1CBANDS.index(b) for b in L2ABANDS]]

        if window is not None:
            win_transform = src.window_transform(window)
        else:
            win_transform = src.transform
    return image, win_transform


bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def calculate_fdi(scene):
    # scene values [0,1e4]

    NIR = scene[bands.index("B8")] * 1e-4
    RED2 = scene[bands.index("B6")] * 1e-4
    SWIR1 = scene[bands.index("B11")] * 1e-4

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    img = NIR - NIR_prime
    return img

def calculate_ndvi(scene):
    NIR = scene[bands.index("B8")] * 1e-4
    RED = scene[bands.index("B4")] * 1e-4
    img = (NIR - RED) / (NIR + RED + 1e-12)
    return img

def s2_to_rgb(scene):
    tensor = np.stack([scene[bands.index('B4')],scene[bands.index('B3')],scene[bands.index('B2')]])
    return equalize_hist(tensor.swapaxes(0,1).swapaxes(1,2))

def rgb(scene):
    return equalize_hist(scene[np.array([3,2,1])])