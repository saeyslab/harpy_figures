# Figures for the Harpy manuscript

This repository contains the code to reproduce the figures in the Harpy manuscript and uses the [Harpy package](https://github.com/saeyslab/harpy).

## Install
This code is Linux only. Install [Pixi](https://pixi.sh/latest/).

```
pixi i -e all
```

## Activate environment
```
pixi shell -e harpy
```

## Usage

### Setup datasets

Each script will also create a dataset based on code in the harpy package if not present. For the aggregation task, you can use the dataset created by the segmentation task by moving it to the location `$ORIG_DATASET` as seen in script `submit_aggregation_harpy_jobs.sh`. Creating the largest datasets can take a while, can take several Gigabytes count for the Slurm job limit. Each dataset is removed after a successful run.

### Run the tasks

The following commands will execute for a certain comparison task several jobs on a Slurm cluster like e.g. [doduo on HPC-UGent](https://docs.hpc.ugent.be/Linux/infrastructure/). To have multiple measurements per task, you can run the same command e.g. 3 times. The results will be saved in the `hpc_scripts/.duct` folder.

Make sure to update the `$VERSION` variable in the scripts for each run.

```bash
ml load cluster/doduo
cd hpc_scripts
bash submit_segment_jobs.sh # submit the segmentation jobs
bash submit_aggregation_harpy_jobs.sh # submit the aggregation jobs
bash submit_flowsom_harpy_jobs.sh # submit the flowsom jobs
```

### Create the figures with notebooks

The figures are created with the Jupyter notebooks in the `figures` folder. You can run them in a Jupyter notebook in the Pixi environment.

- `notebooks/process_tonsil.ipynb`
    - Figure 2 (A, D)
- `notebooks/figures_segmentation.ipynb`
    - Figure 3 (I)
- `notebooks/figures_segmentation_quality.ipynb`
    - Figure 3 (A, B, C, D, E, F)
- `notebooks/figures_agg.ipynb`
    - Figure 4 (A, B)
- `notebooks/figures_cluster.ipynb`
    - Figure 5 (A, B)
- `notebooks/figures_cluster_quality.ipynb`
    - Figure 5 (C, D)

They use code from the `helpers.py` script to postprocess the output in `hpc_scripts/.duct`. Make sure to update the paths in the notebooks to point to the correct version location of the output files.

See the [Harpy documentation](https://harpy.readthedocs.io/en/latest/) for more information on specific package functionality and plots.