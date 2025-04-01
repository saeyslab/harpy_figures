#!/usr/bin/env python3
import argparse
import harpy as hp

from pathlib import Path

from spatialdata import read_zarr

import dask.array as da
import math
import numpy as np


def create_cellpose_dataset(
    path: str | Path,
    y_dim: int = 10000,
    x_dim: int = 10000,
    chunksize: int = 4096,
    img_layer: str = "image_tiled",
    dtype: str = np.float32,  # sopa only accepts np.uint
):
    # Generate the example spatial data
    sdata = hp.datasets.vectra_example()

    sdata.write(path, overwrite=False)

    sdata = read_zarr(sdata.path)

    orig_shape = sdata["image"].data.shape

    # Compute the minimum number of repetitions required along each axis
    repeat_y = math.ceil(y_dim / orig_shape[1])
    repeat_x = math.ceil(x_dim / orig_shape[2])

    # Tile the array
    tiled = da.tile(sdata["image"].data, (1, repeat_y, repeat_x))

    # take channel at index 6 and 8, this is a DAPI stain, and some cell specific stain.
    tiled = tiled[6:8, ...]

    tiled = tiled[:, :y_dim, :x_dim].rechunk((tiled.shape[0], chunksize, chunksize))

    hp.im.add_image_layer(
        sdata, arr=tiled.astype(dtype), output_layer=img_layer, overwrite=True
    )

    return sdata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate vectra example spatial data and write it to a zarr file."
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="The output path where the zarr file will be saved.",
    )
    parser.add_argument(
        "--y_dim",
        type=int,
        default=10000,
        help="The target size along the y-axis (second axis). Default is 10000.",
    )
    parser.add_argument(
        "--x_dim",
        type=int,
        default=10000,
        help="The target size along the x-axis (third axis). Default is 10000.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4096,
        help="Chunksize in y and x. Chunksize in c is c_dim.",
    )
    args = parser.parse_args()
    create_cellpose_dataset(
        path=args.output_path,
        y_dim=args.y_dim,
        x_dim=args.x_dim,
        chunksize=args.chunksize,
    )
