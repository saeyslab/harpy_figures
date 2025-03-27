from pathlib import Path

import dask
import dask.array as da
import loguru
import spatialdata as sd

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

    return dfs


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

    return result


def spatialdata_aggregation(
    sdata: sd.SpatialData,
    img_layer: str,
    labels_layer: str,
    workers: int | None = None,
    threads: int | None = None,
):
    from dask.distributed import Client, LocalCluster

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

    sdata = sd.aggregate(
        values=sdata[img_layer], by=sdata[labels_layer], agg_func="mean"
    )
    logger.info(
        f"Aggregation done, obtained AnnDAta object of shape '{sdata['table'].shape}'."
    )

    return sdata


def sopa_aggregation(
    sdata: sd.SpatialData,
    img_layer: str,
    labels_layer: str,
    workers: int | None = None,
    threads: int | None = None,
):
    from dask.distributed import Client, LocalCluster
    import harpy as hp
    import sopa

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

    # TODO: do vectorization outside this function
    logger.info("Start vectorization.")
    sdata = hp.sh.vectorize(
        sdata,
        labels_layer=labels_layer,
        output_layer="shapes_cells_harpy",
        overwrite=True,
    )
    logger.info("End vectorization.")

    logger.info("Start aggregation.")

    sopa.aggregate(
        sdata,
        aggregate_genes=False,
        aggregate_channels=True,
        image_key=img_layer,
        shapes_key="shapes_cells_harpy",
        key_added="table",
    )

    logger.info(
        f"Aggregation done, obtained AnnDAta object of shape '{sdata['table'].shape}'."
    )


def squidpy_aggregation(
    sdata: sd.SpatialData,
    img_layer: str,
    labels_layer: str,
    n_jobs: int = 1,
    diameter: int = 100,
):
    import squidpy as sq
    import anndata
    import numpy as np
    from scipy.ndimage import center_of_mass

    logger.info(f"Number of cores used: {n_jobs}.")

    labels = da.unique(sdata[labels_layer].data).compute()
    labels = labels[labels != 0]

    adata = anndata.AnnData(X=np.empty((labels.shape[0], 0)))
    adata.obs_names = [f"cell_{i}" for i in labels]
    adata.obs["library_id"] = "region"
    adata.obs["cell_id"] = labels
    arr_image = sdata[img_layer].data.compute().transpose(1, 2, 0)
    arr_segmentation = sdata[labels_layer].data.compute()

    array_center_of_mass = np.array(
        center_of_mass(input=arr_segmentation, labels=arr_segmentation, index=labels)
    )
    array_center_of_mass = array_center_of_mass[
        :, [1, 0]
    ]  # adata.obsm["spatial"] should be x,y

    dictionary = {"region": {"scalefactors": {"spot_diameter_fullres": diameter}}}

    adata.uns["spatial"] = dictionary
    adata.obsm["spatial"] = array_center_of_mass

    imgs = []
    for library_id in adata.uns["spatial"].keys():
        img = sq.im.ImageContainer(arr_image, library_id=library_id)
        img.add_img(
            arr_segmentation,
            library_id=library_id,
            layer="segmentation",
        )
        img["segmentation"].attrs["segmentation"] = True
        imgs.append(img)
    img = sq.im.ImageContainer.concat(imgs)

    def segmentation_image_intensity(arr, image):
        """
        Calculate per-channel mean intensity of the center segment.

        arr: the segmentation
        image: the raw image values
        """
        import skimage.measure

        # the center of the segmentation mask contains the current label
        # use that to calculate the mask
        s = arr.shape[0]
        mask = (arr == arr[s // 2, s // 2, 0, 0]).astype(int)
        # use skimage.measure.regionprops to get the intensity per channel
        features = []
        for c in range(image.shape[-1]):
            feature = skimage.measure.regionprops_table(
                np.squeeze(
                    mask
                ),  # skimage needs 3d or 2d images, so squeeze excess dims
                intensity_image=np.squeeze(image[:, :, :, c]),
                properties=["mean_intensity"],
            )["mean_intensity"][0]
            features.append(feature)
        return features

    sq.im.calculate_image_features(
        adata,
        img,
        library_id="library_id",
        features="custom",
        spot_scale=1,
        n_jobs=n_jobs,
        layer="segmentation",
        features_kwargs={
            "custom": {
                "func": segmentation_image_intensity,
                "additional_layers": ["image"],
            }
        },
    )

    logger.info(
        f"Aggregation done, obtained image features of shape: {adata.obsm['img_features'].shape}"
    )

    return adata


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

    if args.method == "spatialdata":
        spatialdata_aggregation(
            sdata,
            img_layer=args.img_layer,
            labels_layer=args.labels_layer,
            workers=args.workers,
            threads=args.threads,
        )

    if args.method == "sopa":
        sopa_aggregation(
            sdata,
            img_layer=args.img_layer,
            labels_layer=args.labels_layer,
            workers=args.workers,
            threads=args.threads,
        )

    if args.method == "squidpy":
        squidpy_aggregation(
            sdata,
            img_layer=args.img_layer,
            labels_layer=args.labels_layer,
            n_jobs=args.workers * args.threads,
        )
