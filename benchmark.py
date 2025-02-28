import os
from pathlib import Path

import loguru
import numpy as np
import spatialdata as sd

from prep_multi_channel_dataset import create_multi_channel_dataset

logger = loguru.logger


def harpy_segment(
    sdata: sd.SpatialData,
    chunksize: int,
    img_layer: str,
    workers: int | None = None,
    threads: int | None = None,
):
    import harpy as hp
    from dask.distributed import Client, LocalCluster
    from instanseg import InstanSeg

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
            "Workers or threads not specified, running segmentation without a client."
        )

    logger.info("Start segmentation.")

    _ = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1, device="cpu")

    path_model = os.path.join(
        os.environ.get("INSTANSEG_BIOIMAGEIO_PATH"),
        "fluorescence_nuclei_and_cells/0.1.0/instanseg.pt",
    )

    logger.info(f"Path to instanseg model: {path_model}.")

    sdata = hp.im.segment(
        sdata,
        img_layer=img_layer,
        output_labels_layer=["labels_nuclei_harpy", "labels_cells_harpy"],
        output_shapes_layer=None,
        labels_layer_align=None,
        chunks=(chunksize, chunksize),
        depth=50,
        model=hp.im.instanseg_callable,
        # parameters passed to hp.im.instanseg_callable
        output="all_outputs",
        device="cpu",
        instanseg_model=path_model,  # load it in every worker, because torchscript model is not serializable
        iou=True,
        trim=False,
        overwrite=True,
    )


def instanseg_segment(
    sdata: sd.SpatialData,
    chunksize: int,
    img_layer: str,
):
    import torch
    from instanseg import InstanSeg

    _ = InstanSeg("fluorescence_nuclei_and_cells", verbosity=1, device="cpu")

    path_model = os.path.join(
        os.environ.get("INSTANSEG_BIOIMAGEIO_PATH"),
        "fluorescence_nuclei_and_cells/0.1.0/instanseg.pt",
    )

    instanseg_model = torch.load(path_model)
    instanseg_model = InstanSeg(model_type=instanseg_model, device="cpu")

    image_array = sdata[img_layer].data.compute()

    labeled_output, _ = instanseg_model.eval_medium_image(
        image_array,
        tile_size=chunksize,
        batch_size=1,
        resolve_cell_and_nucleus=True,
        cleanup_fragments=True,
        target="all_outputs",
    )  # "all_outputs", "nuclei", or "cells".

    nuclei, cells = labeled_output.squeeze(0).numpy().astype(np.uint32)
    sdata["labels_nuclei_instanseg"] = sd.models.Labels2DModel.parse(
        nuclei, dims=("y", "x")
    )
    sdata.write_element("labels_nuclei_instanseg")
    sdata["labels_cells_instanseg"] = sd.models.Labels2DModel.parse(
        cells, dims=("y", "x")
    )
    sdata.write_element("labels_cells_instanseg")


def sopa_segment(sdata, layer="blobs_image", channel="nucleus", jobs=None):
    import sopa

    if jobs:
        sopa.settings.parallelization_backend = "dask"
        sopa.settings.dask_client_kwargs["n_workers"] = jobs
    sopa.make_image_patches(sdata, image_key="blobs_image")
    sopa.segmentation.cellpose(
        sdata, channels=[channel], diameter=10, image_key="blobs_image"
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
        help="Path where a multi channel artificial dataset will be created.",
    )
    parser.add_argument(
        "--c_dim",
        type=int,
        default=10,
        help="The target number of channels (first axis). Default is 10. Ignored if dataset already exists.",
    )
    parser.add_argument(
        "--y_dim",
        type=int,
        default=10000,
        help="The target size along the y-axis (second axis). Default is 10000. Ignored if dataset already exists.",
    )
    parser.add_argument(
        "--x_dim",
        type=int,
        default=10000,
        help="The target size along the x-axis (third axis). Default is 10000. Ignored if dataset already exists.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=4096,
        help="Chunksize in y and x. Chunksize in c is c_dim.",
    )
    parser.add_argument(
        "--img_layer",
        type=str,
        default="image_tiled",
        help="Name of the image layer in spatialdata object to run benchmark on.",
    )
    parser.add_argument("--method", help="Method to use", default="harpy")
    parser.add_argument("--threads", help="Threads per worker", default=None, type=int)
    parser.add_argument("--workers", help="Workers to use", default=None, type=int)

    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    # p_profile: Path = Path(args.profile).resolve()
    # if not p_profile.parent.exists():
    #     p_profile.parent.mkdir(parents=True)
    if not d.exists():
        logger.info(f"Dataset {d} does not exist. Creating dataset at {args.dataset}")
        sdata = create_multi_channel_dataset(
            path=args.dataset,
            c_dim=args.c_dim,
            y_dim=args.y_dim,
            x_dim=args.x_dim,
            chunksize=args.chunksize,
            img_layer=args.img_layer,
        )
    else:
        raise FileExistsError(
            f"A dataset already exists at {d}. To create a new dataset, "
            "please specify a different path or remove the existing dataset."
        )
        # logger.info(f"Dataset already exists. Reading dataset at {args.dataset}")
        # sdata = sd.read_zarr(d)
    # sdata needs to be backed, otherwise we persist mask in memory
    # sdata.path = None
    if args.method == "harpy":
        harpy_segment(
            sdata,
            chunksize=args.chunksize,
            img_layer=args.img_layer,
            workers=args.workers,
            threads=args.threads,
        )
    if args.method == "instanseg":
        instanseg_segment(
            sdata,
            chunksize=args.chunksize,
            img_layer=args.img_layer,
        )
    if args.method == "sopa":
        sopa_segment(sdata)
    elif args.method == "sopa_dask":
        sopa_segment(sdata, jobs=args.threads)
