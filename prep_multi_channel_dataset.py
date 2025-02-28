#!/usr/bin/env python3
import argparse
import harpy as hp

from pathlib import Path

from spatialdata import read_zarr

import dask.array as da
import math


def create_multi_channel_dataset(
    path: str | Path,
    c_dim: int = 10,
    y_dim: int = 10000,
    x_dim: int = 10000,
    chunksize: int = 4096,
):
    # Generate the example spatial data
    sdata = hp.datasets.vectra_example()

    sdata.write(path, overwrite=False)

    sdata = read_zarr(sdata.path)

    orig_shape = sdata["image"].data.shape

    # Compute the minimum number of repetitions required along each axis
    repeat_z = math.ceil(c_dim / orig_shape[0])
    repeat_y = math.ceil(y_dim / orig_shape[1])
    repeat_x = math.ceil(x_dim / orig_shape[2])

    # Tile the array
    tiled = da.tile(sdata["image"].data, (repeat_z, repeat_y, repeat_x))

    tiled = tiled[:c_dim, :y_dim, :x_dim].rechunk((c_dim, chunksize, chunksize))

    hp.im.add_image_layer(sdata, arr=tiled, output_layer="image_tiled", overwrite=True)


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
        "--c_dim",
        type=int,
        default=10,
        help="The target number of channels (first axis). Default is 10.",
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
    create_multi_channel_dataset(
        path=args.output_path,
        c_dim=args.c_dim,
        y_dim=args.y_dim,
        x_dim=args.x_dim,
        chunksize=args.chunksize,
    )
