from pathlib import Path

import loguru

logger = loguru.logger

def create_dataset(p: Path, **kwargs):
    import harpy

    # create a dataset
    sdata = harpy.datasets.cluster_blobs(**kwargs)
    logger.info(f"Created dataset {sdata}")
    sdata.write(p)

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    p_data = root / "data"
    logger.info(f"Creating datasets in {p_data}")
    if p_data.exists():
        logger.info(f"Removing existing datasets in {p_data}")
        import shutil
        shutil.rmtree(p_data)
    p_data.mkdir()

    # create a harpy
    for i in range(3):
        p = p_data / f"dataset_{i}.zarr"
        size = 100*(i+1)
        logger.info(f"Creating dataset {p} with size {size}")
        create_dataset(p, shape=(size, size), n_cells=10**i, n_cell_types=10)
        