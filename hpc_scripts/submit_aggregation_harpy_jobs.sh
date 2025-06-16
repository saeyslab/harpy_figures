#!/bin/bash
# Define your parameter arrays:
DIM_VALUES=(500 1000 2000 5000 10000 20000)
# DIM_VALUES=(500)
RAM_VALUES=(100 100 100 100 100 100)  # Memory in GB
# RAM_VALUES=(10)
TIME_VALUES=(04 04 04 04 04 04)
# TIME_VALUES=(01)

WORKERS_VALUES=(1 2 4 8)
# WORKERS_VALUES=(1)
METHOD_VALUES=(harpy xr_spatial spatialdata sopa squidpy)
ENV_VALUES=(harpy harpy harpy sopa squidpy)
VERSION=014

for i in "${!DIM_VALUES[@]}"; do
  DIM=${DIM_VALUES[$i]}
  MEM=${RAM_VALUES[$i]}
  TIME=${TIME_VALUES[$i]}
  for WORKERS in "${WORKERS_VALUES[@]}"; do
    for j in "${!METHOD_VALUES[@]}"; do
      METHOD=${METHOD_VALUES[$j]}
      ENV=${ENV_VALUES[$j]}
      # Create the job script
      sbatch <<EOF
#!/bin/bash
#SBATCH --partition=doduo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${WORKERS}
#SBATCH --time=${TIME}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --job-name=aggregation_${VERSION}_${METHOD}_${DIM}_${WORKERS}
#SBATCH --output=aggregation_${VERSION}_${METHOD}_${DIM}_${WORKERS}.out
#SBATCH --error=aggregation_${VERSION}_${METHOD}_${DIM}_${WORKERS}.err

# Set parameters
DIM=${DIM}
C_DIM=20
THREADS=${WORKERS}
CHUNKSIZE=1000
WORKERS=1
METHOD=${METHOD}
VERSION=${VERSION}
ENV=${ENV}
ORIG_DATASET="\$VSC_DATA_VO_USER/VIB/DATA/sdata_segment_011_harpy_\${DIM}_8.zarr"
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_agg_\${VERSION}_\${METHOD}_\${DIM}_\${THREADS}.zarr"
IMG_LAYER="image_tiled"
LABELS_LAYER="labels_cells_harpy"
LOG_DIR=".duct/logs_aggregation_\${VERSION}"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${DIM}_\${THREADS}"

mkdir -p "\$LOG_DIR"

# copy the dataset to a local directory
cp -r "\$ORIG_DATASET" "\$DATASET"

pixi run --frozen -e "\$ENV" duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_aggregation.py \\
  --dataset "\$DATASET" \\
  --img_layer "\$IMG_LAYER" \\
  --labels_layer  "\$LABELS_LAYER" \\
  --threads "\$THREADS" \\
  --workers 1 \\
  --method "\$METHOD"

rm -r "\$DATASET"
EOF
  done
done
done