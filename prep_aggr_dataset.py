from pathlib import Path

def prep_dataset(sdata, labels_layer='labels_cells_harpy'):
    import harpy
    harpy.sh.vectorize(
        sdata,
        labels_layer=labels_layer,
        output_layer="shapes_cells_harpy",
        overwrite=True,
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

    args = parser.parse_args()
    # load dataset
    import spatialdata as sd

    p = args.dataset
    if not p.exists():
        raise FileNotFoundError(f"Dataset {args.dataset} does not exist.")
    sdata = sd.read_zarr(p)
    # prep dataset
    prep_dataset(sdata)
    print("Dataset prep complete.")