import os
import random

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import torch
from rasterio import features
from rasterio.windows import from_bounds
from shapely.geometry import LineString
from shapely.geometry import Polygon

l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET = 1000
allregions = [
    "accra_20181031",
    "biscay_20180419",
    "danang_20181005",
    "kentpointfarm_20180710",
    "kolkata_20201115",
    "lagos_20190101",
    "lagos_20200505",
    "london_20180611",
    "longxuyen_20181102",
    "mandaluyong_20180314",
    "neworleans_20200202",
    "panama_20190425",
    "portalfredSouthAfrica_20180601",
    "riodejaneiro_20180504",
    "sandiego_20180804",
    "sanfrancisco_20190219",
    "shengsi_20190615",
    "suez_20200403",
    "tangshan_20180130",
    "toledo_20191221",
    "tungchungChina_20190922",
    "tunisia_20180715",
    "turkmenistan_20181030",
    "venice_20180630",
    "venice_20180928",
    "vungtau_20180423"
    ]


def get_region_split(seed=0, fractions=(0.6, 0.2, 0.2)):
    # fix random state
    random_state = np.random.RandomState(seed)

    # shuffle sequence of regions
    shuffled_regions = random_state.permutation(allregions)

    # determine first N indices for training
    train_idxs = np.arange(0, np.floor(len(shuffled_regions) * fractions[0]).astype(int))

    # next for validation
    idx = np.ceil(len(shuffled_regions) * (fractions[0] + fractions[1])).astype(int)
    val_idxs = np.arange(np.max(train_idxs) + 1, idx)

    # the remaining for test
    test_idxs = np.arange(np.max(val_idxs) + 1, len(shuffled_regions))

    return dict(train=list(shuffled_regions[train_idxs]),
                validation=list(shuffled_regions[val_idxs]),
                test=list(shuffled_regions[test_idxs]))

def line_is_closed(linestringgeometry):
    coordinates = np.stack(linestringgeometry.xy).T
    first_point = coordinates[0]
    last_point = coordinates[-1]
    return bool((first_point == last_point).all())

def split_line_gdf_into_segments(lines):
    def segments(curve):
        return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))

    line_segments = []
    for geometry in lines.geometry:
        line_segments += segments(geometry)
    return gpd.GeoDataFrame(geometry=line_segments)

def random_crop(image, mask, cropsize):
    C, W, H = image.shape
    w, h = cropsize, cropsize

    # distance from image border
    dh, dw = h // 2, w // 2

    # sample some point inside the valid square
    x = np.random.randint(dw, W - dw)
    y = np.random.randint(dh, H - dh)

    # crop image
    image = image[:, x - dw:x + dw, y - dh:y + dh]
    mask = mask[:, x - dw:x + dw, y - dh:y + dh]

    return image, mask

def get_data_augmentation(intensity):
    """
    do data augmentation:
    model
    """
    def data_augmentation(image, mask):
        image = torch.Tensor(image)
        mask = torch.Tensor(mask)
        mask = mask.unsqueeze(0)

        if random.random() < 0.5:
            # flip left right
            image = torch.fliplr(image)
            mask = torch.fliplr(mask)

        rot = np.random.choice([0,1,2,3])
        image = torch.rot90(image, rot, [1, 2])
        mask = torch.rot90(mask, rot, [1, 2])

        if random.random() < 0.5:
            # flip up-down
            image = torch.flipud(image)
            mask = torch.flipud(mask)

        if intensity >= 1:

            # random crop
            cropsize = image.shape[2] // 2
            image, mask = random_crop(image, mask, cropsize=cropsize)

            std_noise = 1 * image.std()
            if random.random() < 0.5:
                # add noise per pixel and per channel
                pixel_noise = torch.rand(image.shape[1], image.shape[2])
                pixel_noise = torch.repeat_interleave(pixel_noise.unsqueeze(0), image.size(0), dim=0)
                image = image + pixel_noise*std_noise

            if random.random() < 0.5:
                channel_noise = torch.rand(image.shape[0]).unsqueeze(1).unsqueeze(2)
                channel_noise = torch.repeat_interleave(torch.repeat_interleave(channel_noise, image.shape[1], 1),
                                                        image.shape[2], 2)
                image = image + channel_noise*std_noise

            if random.random() < 0.5:
                # add noise
                noise = torch.rand(image.shape[0], image.shape[1], image.shape[2]) * std_noise
                image = image + noise

        if intensity >= 2:
            # channel shuffle
            if random.random() < 0.5:
                idxs = np.arange(image.shape[0])
                np.random.shuffle(idxs) # random band indixes
                image = image[idxs]

        mask = mask.squeeze(0)
        mask = torch.Tensor(np.expand_dims(mask, axis=0))
        return image, mask
    return data_augmentation

def calculate_fdi(scene):
    # scene values [0,1e4]

    NIR = scene[l2abands.index("B8")] * 1e-4
    RED2 = scene[l2abands.index("B6")] * 1e-4
    SWIR1 = scene[l2abands.index("B11")] * 1e-4

    lambda_NIR = 832.9
    lambda_RED = 664.8
    lambda_SWIR1 = 1612.05
    NIR_prime = RED2 + (SWIR1 - RED2) * 10 * (lambda_NIR - lambda_RED) / (lambda_SWIR1 - lambda_RED)

    img = NIR - NIR_prime
    return img

def calculate_ndvi(scene):
    NIR = scene[l2abands.index("B8")] * 1e-4
    RED = scene[l2abands.index("B4")] * 1e-4
    img = (NIR - RED) / (NIR + RED + 1e-12)
    return img

def get_transform(mode, intensity=0, add_fdi_ndvi=False):
    assert mode in ["train", "test"]
    if mode in ["train"]:
        def train_transform(image, mask):

            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4
            # return image, mask
            data_augmentation = get_data_augmentation(intensity=intensity)
            return data_augmentation(image, mask)
        return train_transform
    else:
        def test_transform(image, mask):
            if add_fdi_ndvi:
                fdi = np.expand_dims(calculate_fdi(image),0)
                ndvi = np.expand_dims(calculate_ndvi(image),0)
                image = np.vstack([image,ndvi,fdi])

            image *= 1e-4
            image = torch.Tensor(image)
            mask = torch.Tensor(np.expand_dims(mask, axis=0))
            return image, mask
        return test_transform

class FloatingSeaObjectRegionDataset(torch.utils.data.Dataset):
    def __init__(self, root, region, output_size=64,
                 channels=12, hard_negative_mining=False,
                 use_l2a_probability=0.5):

        shapefile = os.path.join(root, region + ".shp")
        imagefile = os.path.join(root, region + ".tif")
        imagefilel2a = os.path.join(root, region + "_l2a.tif")

        # if 0.5 use 50% of time L2A image (if available)
        # if 0 only L1C images are used
        # if 1 only L2A images are used
        self.use_l2a_probability = 0.5

        # return zero-element dataset if use_l2a_probability=1 but l2a file not available
        if use_l2a_probability == 1 and not os.path.exists(imagefilel2a):
            self.lines = []
            return  # break early out of this function

        self.transform = get_transform("train", intensity=0, add_fdi_ndvi=(channels == 14))
        self.region = region

        self.imagefile = imagefile
        self.imagefilel2a = imagefilel2a
        self.output_size = output_size

        with rio.open(imagefile) as src:
            self.imagemeta = src.meta
            self.imagebounds = tuple(src.bounds)

        lines = gpd.read_file(shapefile)
        lines = lines.to_crs(self.imagemeta["crs"])

        # find closed lines, convert them to polygons and store them separately for later rasterization
        is_closed_line = lines.geometry.apply(line_is_closed)
        rasterize_polygons = lines.loc[is_closed_line].geometry.apply(Polygon)

        self.lines = split_line_gdf_into_segments(lines)

        self.lines["is_hnm"] = False
        if hard_negative_mining:
            random_points = self.sample_points_for_hard_negative_mining()
            random_points["is_hnm"] = True
            self.lines = pd.concat([self.lines, random_points]).reset_index(drop=True)

        # remove line segments that are outside the image bounds
        self.lines = self.lines.loc[self.lines.geometry.apply(self.within_image)]

        # take lines to rasterize
        rasterize_lines = self.lines.loc[~self.lines["is_hnm"]].geometry

        # combine with polygons to rasterize
        self.rasterize_geometries = pd.concat([rasterize_lines, rasterize_polygons])

    def within_image(self, geometry):
        left, bottom, right, top = geometry.bounds
        ileft, ibottom, iright, itop = self.imagebounds
        return ileft < left and iright > right and itop > top and ibottom < bottom

    def sample_points_for_hard_negative_mining(self):
        # hard negative mining:
        # get some random negatives from the image bounds to ensure that the model can learn on negative examples
        # e.g. land, clouds, etc

        with rio.open(self.imagefile) as src:
            left, bottom, right, top = src.bounds

        offset = HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET  # m
        assert top - bottom > 2 * HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET, f"Hard Negative Mining offset 2x{HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET}m too large for the image height: {top - bottom}m"
        assert right - left > 2 * HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET, f"Hard Negative Mining offset 2x{HARD_NEGATIVE_MINING_SAMPLE_BORDER_OFFSET}m too large for the image width: {right - left}m"
        N_random_points = len(self.lines)

        # sample random x positions within bounds
        zx = np.random.rand(N_random_points)
        zx *= ((right - offset) - (left + offset))
        zx += left + offset

        # sample random y positions within bounds
        zy = np.random.rand(N_random_points)
        zy *= ((top - offset) - (bottom + offset))
        zy += bottom + offset

        return gpd.GeoDataFrame(geometry=gpd.points_from_xy(zx, zy))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines.iloc[index]
        left, bottom, right, top = line.geometry.bounds

        width = right - left
        height = top - bottom

        # buffer_left_right = (self.output_size[0] * 10 - width) / 2
        buffer_left_right = (self.output_size * 10 - width) / 2
        left -= buffer_left_right
        right += buffer_left_right

        # buffer_bottom_top = (self.output_size[1] * 10 - height) / 2
        buffer_bottom_top = (self.output_size * 10 - height) / 2
        bottom -= buffer_bottom_top
        top += buffer_bottom_top

        window = from_bounds(left, bottom, right, top, self.imagemeta["transform"])

        imagefile = self.imagefile

        if os.path.exists(self.imagefilel2a):
            if np.random.rand() > self.use_l2a_probability:
                imagefile = self.imagefilel2a

        with rio.open(imagefile) as src:
            image = src.read(window=window)
            # keep only 12 bands: delete 10th band (nb: 9 because start idx=0)
            if (image.shape[0] == 13):  # is L1C Sentinel 2 data
                image = image[[l1cbands.index(b) for b in l2abands]]

            win_transform = src.window_transform(window)

        h_, w_ = image[0].shape
        assert h_ > 0 and w_ > 0, f"{self.region}-{index} returned image size {image[0].shape}"
        # only rasterize the not-hard negative mining samples

        mask = features.rasterize(self.rasterize_geometries, all_touched=True,
                                  transform=win_transform, out_shape=image[0].shape)

        # if feature is near the image border, image wont be the desired output size
        H, W = self.output_size, self.output_size
        c, h, w = image.shape
        dh = (H - h) / 2
        dw = (W - w) / 2
        image = np.pad(image, [(0, 0), (int(np.ceil(dh)), int(np.floor(dh))),
                               (int(np.ceil(dw)), int(np.floor(dw)))])

        mask = np.pad(mask, [(int(np.ceil(dh)), int(np.floor(dh))),
                             (int(np.ceil(dw)), int(np.floor(dw)))])

        mask = mask.astype(float)
        image = image.astype(float)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        assert not np.isnan(image).any()
        assert not np.isnan(mask).any()

        return image, mask, index


class sentinel2(torch.utils.data.ConcatDataset):
    def __init__(self, root, fold="train", seed=0, **kwargs):
        assert fold in ["train", "validation", "test"]

        # make regions variable available to the outside
        self.regions = get_region_split(seed)[fold]

        # initialize a concat dataset with the corresponding regions
        super().__init__(
            [FloatingSeaObjectRegionDataset(root, region, **kwargs) for region in self.regions]
        )