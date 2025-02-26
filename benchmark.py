from pathlib import Path

import loguru
import spatialdata as sd

logger = loguru.logger



def run_on_dataset(sdata):
    import harpy

    logger.info(f"Running on dataset {sdata}")
    harpy.im.normalize(sdata, img_layer='blobs_image', output_layer='norm_image')
    logger.info(f"Normalized dataset {sdata}")


if __name__ == "__main__":
    import argparse

    import memray

    # get path from argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset")
    args = parser.parse_args()
    d = Path(args.dataset).resolve()
    # p_profile: Path = Path(args.profile).resolve()
    # if not p_profile.parent.exists():
    #     p_profile.parent.mkdir(parents=True)
    assert d.exists(), f"Dataset {d} does not exist"

    sdata = sd.read_zarr(d)
    # make sure no changes are written back
    sdata.path = None
    run_on_dataset(sdata)
