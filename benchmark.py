from pathlib import Path

import loguru
import spatialdata as sd

logger = loguru.logger


def harpy_segment(sdata, layer="blobs_image", channel='nucleus', jobs=None):
    import harpy
    from dask.distributed import LocalCluster


    logger.info(f"Running on dataset {sdata}")
    if jobs:
        cluster = LocalCluster(
            # See [LocalCluster docs](https://distributed.dask.org/en/stable/api.html#cluster)
            # the number of workers to start
            n_workers=jobs,
            # the number of threads per worker, set to 1 to avoid oversubscription and only use Dask for parallelisation
            threads_per_worker=1,
            # the hard memory limit for *every* worker
            # memory_limit="4GB",
            host="127.0.0.1",
            # see [Worker API docs](https://distributed.dask.org/en/stable/worker.html#api-documentation)
        )
        _ = cluster.get_client()
    sdata=harpy.im.add_image_layer( sdata, arr=sdata[ layer ].sel( c=channel ).data[ None, ... ], output_layer="_image_", overwrite=True  )
    sdata_seg = harpy.im.segment(sdata, img_layer="_image_", depth=10)

def sopa_segment(sdata, layer="blobs_image", channel='nucleus', jobs=None):
    import sopa

    if jobs:
        sopa.settings.parallelization_backend = 'dask'
        sopa.settings.dask_client_kwargs["n_workers"] = jobs
    sopa.make_image_patches(sdata, image_key="blobs_image")
    sopa.segmentation.cellpose(sdata, channels=[channel], diameter=10, image_key="blobs_image")

if __name__ == "__main__":
    import argparse

    # get path from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Path to dataset")
    parser.add_argument("--method", help="Method to use", default="harpy")
    parser.add_argument("--threads", help="Threads to use", default=None, type=int)
    parser.add_argument("--memory_per_worker", help="Memory per worker to use", default="4GB")
    parser.add_argument("--gpus", help="GPUs to use", default=None)
    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    # p_profile: Path = Path(args.profile).resolve()
    # if not p_profile.parent.exists():
    #     p_profile.parent.mkdir(parents=True)
    assert d.exists(), f"Dataset {d} does not exist"

    sdata = sd.read_zarr(d).subset("blobs_image")
    # make sure no changes are written back
    sdata.path = None
    if args.method == "harpy":
        harpy_segment(sdata, jobs=args.threads)
    if args.method == "sopa":
        sopa_segment(sdata)
    elif args.method == "sopa_dask":
        sopa_segment(sdata, jobs=args.threads)
