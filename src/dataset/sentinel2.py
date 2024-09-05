import torch
from rasterio.windows import from_bounds
import rasterio as rio
from rasterio import features
from shapely.geometry import LineString, Polygon
import geopandas as gpd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


l1cbands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
l2abands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

allregions = [
    # "accra_20181031",
    # "biscay_20180419",
    # "danang_20181005",
    # "kentpointfarm_20180710",
    # "kolkata_20201115",
    # "lagos_20190101",
    # "lagos_20200505",
    # "london_20180611",
    # "longxuyen_20181102",
    # "mandaluyong_20180314",
    # "neworleans_20200202",
    # "panama_20190425",
    # "portalfredSouthAfrica_20180601",
    # "riodejaneiro_20180504",
    # "sandiego_20180804",
    # "sanfrancisco_20190219",
    # "shengsi_20190615",
    # "suez_20200403",
    # "tangshan_20180130",
    # "toledo_20191221",
    # "tungchungChina_20190922",
    # "tunisia_20180715",
    # "turkmenistan_20181030",
    "venice_20180630",
    # "venice_20180928",
    # "vungtau_20180423"
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
                val=list(shuffled_regions[val_idxs]),
                test=list(shuffled_regions[test_idxs]))

def line_is_closed(linestringgeometry):
    coordinates = np.stack(linestringgeometry.xy).T
    first_point = coordinates[0]
    last_point = coordinates[-1]
    return bool((first_point == last_point).all())


class FSODataset(torch.utils.data.Dataset):
    def __init__(self, root, fold, seed, output_size, transform, use_l2a_probability):

        self.regions = get_region_split(seed)[fold]

        self.shapefiles = [os.path.join(root, region + '.shp') for region in self.regions]
        self.imagefiles = [os.path.join(root, region + '.tif') for region in self.regions]
        self.l2aimagefiles = [os.path.join(root, region + '._l2a.tif') for region in self.regions]
        self.output_size = output_size

        self.crops = []
        for idx, imagefile in enumerate(self.imagefiles):
            with rio.open(imagefile) as src:
                w, h = src.width, src.height
                n_crops_w = w // output_size
                border_w = (w - output_size * n_crops_w) // 2

                n_crops_h = h // output_size
                border_h = (h - output_size * n_crops_h) // 2
                #for x in range(border_w, w - output_size, output_size):
                #    for y in range(border_h, h - output_size, output_size):
                #        self.crops.append((idx, x, y))
                for x in range(n_crops_w):
                    for y in range(n_crops_h):
                        self.crops.append((idx, border_w + output_size * x, border_h + output_size * y))
                src.close()

        self.use_l2a_probability = use_l2a_probability
        self.transform = transform

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, idx):
        image_id, window_w, window_h = self.crops[idx]
        shapefile = self.shapefiles[image_id]
        imagefile = self.imagefiles[image_id]
        l2aimagefile = self.l2aimagefiles[image_id]

        if os.path.exists(l2aimagefile):
            if np.random.rand() < self.use_l2a_probability:
                imagefile = l2aimagefile

        with rio.open(imagefile) as src:
            imagemeta = src.meta
            imagebounds = tuple(src.bounds)

            # Generate a random window origin (upper left) that ensures the window
            # doesn't go outside the image. i.e. origin can only be between
            # 0 and image width or height less the window width or height
            #xmin, xmax = 0, src.width - self.output_size
            #ymin, ymax = 0, src.height - self.output_size
            #xoff, yoff = np.random.randint(xmin, xmax), np.random.randint(ymin, ymax)

            # Create a Window and calculate the transform from the source dataset
            #window = rio.windows.Window(xoff, yoff, self.output_size, self.output_size)
            window = rio.windows.Window(window_w, window_h, self.output_size, self.output_size)

            win_transform = src.window_transform(window)
            image = src.read(window=window)

            if image.shape[0] == 13:  # is L1C Sentinel 2 dataset
                image = image[[l1cbands.index(b) for b in l2abands]]

            """
            rgb = image[[3, 2, 1], :, :]
            from skimage.exposure import equalize_hist
            plt.imshow(equalize_hist(np.moveaxis(rgb, 0, -1)))
            plt.show()
            """

        def within_image(geometry):
            left, bottom, right, top = geometry.bounds
            ileft, ibottom, iright, itop = imagebounds
            return ileft < left and iright > right and itop > top and ibottom < bottom

        lines = gpd.read_file(shapefile)
        lines = lines.to_crs(imagemeta["crs"])

        is_closed_line = lines.geometry.apply(line_is_closed)
        rasterize_polygons = lines.loc[is_closed_line].geometry.apply(Polygon)
        lines = lines.loc[lines.geometry.apply(within_image)]
        rasterize_lines = lines.geometry
        rasterize_geometries = pd.concat([rasterize_lines, rasterize_polygons])

        mask = features.rasterize(rasterize_geometries, all_touched=True,
                                  transform=win_transform, out_shape=image[0].shape)
        """
        plt.imshow(mask, cmap='gray')
        plt.show()
        """

        mask = mask.astype(float)
        image = image.astype(float)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        mask = np.expand_dims(mask, axis=0)

        return image, mask