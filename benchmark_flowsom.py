from pathlib import Path

import flowsom as fs
import harpy as hp
import loguru
import spatialdata as sd

from prep_multi_channel_dataset import create_multi_channel_macsima_dataset

logger = loguru.logger


def harpy_flowsom(
    sdata: sd.SpatialData,
    img_layer: str,
    workers: int | None = None,
    method: str = "harpy",
    threads: int | None = None,
    batches_flowsom: int = 1,
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
            "Workers or threads not specified, running flowsom clustering without a client."
        )
        client = None

    if method == "flowsom_batch":
        cluster_model = fs.models.BatchFlowSOMEstimator
        cluster_model_kwargs = {
            "num_batches": batches_flowsom,
        }
    elif method == "flowsom":
        cluster_model = fs.models.FlowSOMEstimator
        cluster_model_kwargs = {}
    elif method == "pyflowsom":
        cluster_model = fs.models.PyFlowSOMEstimator
        cluster_model_kwargs = {}
    else:
        raise ValueError(
            f"Method {method} not supported."
        )
    logger.info("Start flowsom pixel clustering.")

    sdata, fsom, mapping = hp.im.flowsom(
        sdata,
        img_layer=[img_layer],
        output_layer_clusters=[
            f"{img_layer}_fov0_flowsom_clusters",
        ],  # we need output_cluster_layer and output_meta_cluster_layer --> these will both be labels layers
        output_layer_metaclusters=[
            f"{img_layer}_fov0_flowsom_metaclusters",
        ],
        n_clusters=20,  # 40
        random_state=112,
        fraction=0.1,
        chunks=512,
        client=client,
        model=cluster_model,
        **cluster_model_kwargs,
        xdim=10,  # 12
        ydim=10,  # 12
        z_score=True,
        z_cap=3,
        persist_intermediate=False,
        overwrite=True,
    )

    logger.info("End flowsom pixel clustering.")


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
    parser.add_argument("--batches_flowsom", help="batches flowsom", default=1, type=int)
    parser.add_argument("--workers", help="Workers to use", default=None, type=int)
    parser.add_argument(
        "--memory_limit", help="memory limit per worker in GB", default=None, type=int
    )
    parser.add_argument(
        "--local_directory",
        type=str,
        default=None,
        help="Local directory where you want to spill to disk.",
    )

    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    if not d.exists():
        logger.info(f"Dataset {d} does not exist. Creating dataset at {args.dataset}")
        sdata = create_multi_channel_macsima_dataset(
            path=args.dataset,
            c_dim=args.c_dim,
            y_dim=args.y_dim,
            x_dim=args.x_dim,
            chunksize=args.chunksize,
            c_chunksize=1,
            img_layer=args.img_layer,
        )
        logger.info("Dataset created. Exiting.")
    else:
        logger.info(
            f"A dataset exists at {d} and will be used."
        )
        sdata = sd.read_zarr(d)
        logger.info("Dataset loaded.")
        harpy_flowsom(
            sdata,
            img_layer=args.img_layer,
            workers=args.workers,
            method=args.method,
            threads=args.threads,
            batches_flowsom=args.batches_flowsom,
            local_directory=args.local_directory,
            memory_limit=args.memory_limit,
        )
