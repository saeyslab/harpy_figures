from pathlib import Path

import harpy as hp
import loguru
import spatialdata as sd

from prep_multi_channel_dataset import create_multi_channel_dataset

logger = loguru.logger


def harpy_preprocess_flowsom(
    sdata: sd.SpatialData,
    img_layer: str,
    workers: int | None = None,
    threads: int | None = None,
    memory_limit: int | None = None,
    local_directory: str
    | Path
    | None = "/kyukon/scratch/gent/vo/001/gvo00163/vsc40523/dask_temp",
):
    from dask.distributed import Client, LocalCluster

    logger.info(
        f"Memory limit per worker is {memory_limit}. Using {workers} workers in total."
    )

    logger.info(f"Running on dataset {sdata}")
    if workers is not None and threads is not None:
        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=threads,
            local_directory=local_directory,
            memory_limit=f"{memory_limit}GB" if memory_limit is not None else None,
        )

        client = Client(cluster)
        logger.info(client.dashboard_link)
    else:
        logger.info(
            "Workers or threads not specified, running preprocessing without a client."
        )

    logger.info("Start preprocessing FlowSOM clustering.")

    sdata = hp.im.pixel_clustering_preprocess(
        sdata,
        img_layer=img_layer,
        output_layer=f"{img_layer}_preprocessed",
        persist_intermediate=False,
        overwrite=True,
    )

    client.close()

    logger.info("End preprocessing FlowSOM clustering.")


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
    parser.add_argument(
        "--img_layer",
        type=str,
        default="image_tiled",
        help="Name of the image layer in spatialdata object to run benchmark on.",
    )
    parser.add_argument("--method", help="Method to use", default="harpy")
    parser.add_argument("--threads", help="Threads per worker", default=None, type=int)
    parser.add_argument("--workers", help="Workers to use", default=None, type=int)
    parser.add_argument(
        "--memory_limit", help="memory limit per worker in GB", default=None, type=int
    )
    parser.add_argument(
        "--local_directory",
        type=str,
        default="/kyukon/scratch/gent/vo/001/gvo00163/vsc40523/dask_temp",
        help="Local directory where you want to spill to disk.",
    )

    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    if not d.exists():
        logger.info(f"Dataset {d} does not exist. Creating dataset at {args.dataset}")
        sdata = create_multi_channel_dataset(
            path=args.dataset,
            c_dim=args.c_dim,
            y_dim=args.y_dim,
            x_dim=args.x_dim,
            chunksize=args.chunksize,
            c_chunksize=1,
            img_layer=args.img_layer,
        )
    else:
        raise FileExistsError(
            f"A dataset already exists at {d}. To create a new dataset, "
            "please specify a different path or remove the existing dataset."
        )

    harpy_preprocess_flowsom(
        sdata,
        img_layer=args.img_layer,
        workers=args.workers,
        threads=args.threads,
        local_directory=args.local_directory,
        memory_limit=args.memory_limit,
    )
