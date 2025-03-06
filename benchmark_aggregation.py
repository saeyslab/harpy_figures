import os
import uuid

from pathlib import Path

import dask
import loguru
import spatialdata as sd
import harpy as hp
import shutil

logger = loguru.logger


def harpy_aggregation(
    sdata: sd.SpatialData,
    img_layer: str,
    labels_layer: str,
    workers: int | None = None,
    threads: int | None = None,
):
    from dask.distributed import Client, LocalCluster
    from harpy.utils._aggregate import RasterAggregator

    logger.info(f"Running on dataset {sdata}")
    if workers is not None and threads is not None:
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads,
            memory_limit="500GB",  # prevent spilling to disk
        )

        client = Client(cluster)
        logger.info(client.dashboard_link)
    else:
        logger.info(
            "Workers or threads not specified, running aggregation without a client."
        )

    logger.info("Start aggregation.")

    image = sdata[img_layer].data[:, None, ...]  # ( "c", "z", "y", "x" )
    labels = sdata[labels_layer].data[None, ...]  # ( "z", "y", "x" )

    aggregator = RasterAggregator(image_dask_array=image, mask_dask_array=labels)
    dfs = aggregator.aggregate_stats(stats_funcs=("mean"))
    logger.info(
        f"Aggregation done, obtained dataframe with mean intensities of shape {dfs[0].shape}"
    )


def xr_spatial_aggregation(
    sdata: sd.SpatialData,
    img_layer: str,
    labels_layer: str,
    workers: int | None = None,
    threads: int | None = None,
):
    from dask.distributed import Client, LocalCluster
    from xrspatial import zonal_stats

    logger.info(f"Running on dataset {sdata}")
    if workers is not None and threads is not None:
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads,
            memory_limit="500GB",  # prevent spilling to disk
        )

        client = Client(cluster)
        logger.info(client.dashboard_link)
    else:
        logger.info(
            "Workers or threads not specified, running aggregation without a client."
        )

    logger.info("Start aggregation.")

    se_image = sdata[img_layer]
    se_labels = sdata[labels_layer]

    ddfs = [
        zonal_stats(
            values=_se_image,
            zones=se_labels,
            stats_funcs=["mean"],
        )
        for _se_image in se_image
    ]

    result = dask.compute(*ddfs)

    logger.info(
        f"Aggregation done, obtained '{len(result)}' dataframes each of shape '{result[0].shape}'."
    )


def zarr_file(value):
    path = Path(value).resolve()
    if path.suffix != ".zarr":
        raise argparse.ArgumentTypeError("Dataset must be a .zarr file.")
    return path


if __name__ == "__main__":
    import argparse

    # get path from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=zarr_file,
        help="Path to dataset.",
    )
    parser.add_argument(
        "--img_layer",
        type=str,
        default="image_tiled",
        help="Name of the image layer in spatialdata object to run benchmark on.",
    )
    parser.add_argument(
        "--labels_layer",
        type=str,
        default="labels_cells_harpy",
        help="Name of the labels layer in spatialdata object to run benchmark on.",
    )
    parser.add_argument("--method", help="Method to use", default="harpy")
    parser.add_argument("--threads", help="Threads per worker", default=None, type=int)
    parser.add_argument("--workers", help="Workers to use", default=None, type=int)

    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    if not d.exists():
        raise FileNotFoundError(f"No .zarr folder found at {d}.")
    else:
        sdata = sd.read_zarr(d)

    if args.method == "harpy":
        harpy_aggregation(
            sdata,
            img_layer=args.img_layer,
            labels_layer=args.labels_layer,
            workers=args.workers,
            threads=args.threads,
        )
    if args.method == "xr_spatial":
        xr_spatial_aggregation(
            sdata,
            img_layer=args.img_layer,
            labels_layer=args.labels_layer,
            workers=args.workers,
            threads=args.threads,
        )
