import numpy as np
from skimage.exposure import equalize_hist


bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
def calculate_fdi(scene):
    # scene values [0,1e4]

    NIR = scene[:,:,bands.index("B8")] * 1e-4
    RED2 = scene[:,:,bands.index("B6")] * 1e-4
    SWIR1 = scene[:,:,bands.index("B11")] * 1e-4

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    img = NIR - NIR_prime
    return img

def calculate_ndvi(scene):
    NIR = scene[:,:,bands.index("B8")] * 1e-4
    RED = scene[:,:,bands.index("B4")] * 1e-4
    img = (NIR - RED) / (NIR + RED + 1e-12)
    return img

def s2_to_rgb(scene):
    tensor = np.stack([scene[:,:,bands.index('B4')],scene[:,:,bands.index('B3')],scene[:,:,bands.index('B2')]],axis=2)
    return equalize_hist(tensor)