from pathlib import Path

import loguru

logger = loguru.logger

def get_root():
    return Path(__file__).resolve()

def create_dataset(p: Path, **kwargs):
    import harpy

    # create a dataset
    sdata = harpy.datasets.cluster_blobs(**kwargs)
    logger.info(f"Created dataset {sdata}")
    sdata.write(p)

if __name__ == "__main__":
    root = get_root()
    p_data = root / "data"
    logger.info(f"Creating datasets in {p_data}")
    p_data.mkdir(exist_ok=True)

    # create a harpy
    for i in range(3):
        p = p_data / f"dataset_{i}.zarr"
        size = 100*(i+1)
        logger.info(f"Creating dataset {p} with size {size}")
        create_dataset(p, shape=(size, size), n_cells=10**i, n_cell_types=10)
        