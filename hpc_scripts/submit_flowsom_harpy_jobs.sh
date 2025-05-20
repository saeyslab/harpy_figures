#!/bin/bash
C_DIM=20
CHUNKSIZE=2048
DIM_VALUES=(500 1000 2000 5000 10000 20000 30000)
# DIM_VALUES=(500)
RAM_VALUES=100  # Memory in GB
TIME_VALUES=4

# for flowsom clustering there are some advantages to run with threads instead of workers, due to sampling.
# WORKERS_VALUES=(1 2)
WORKERS_VALUES=(1 2 4 8)
VERSION=020

for i in "${!DIM_VALUES[@]}"; do
  DIM=${DIM_VALUES[$i]}
  MEM=${RAM_VALUES}
  TIME=${TIME_VALUES}
  for WORKERS in "${WORKERS_VALUES[@]}"; do
    RAM_VALUES_PER_WORKER=$(( MEM ))
    echo $RAM_VALUES_PER_WORKER
    for METHOD in "flowsom" "flowsom_batch" "pyflowsom"; do
      echo $METHOD
      sbatch <<EOF
#!/bin/bash
#SBATCH --partition=doduo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${WORKERS}
#SBATCH --time=${TIME}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --job-name=flowsom_${VERSION}_${DIM}_${WORKERS}_${METHOD}
#SBATCH --output=flowsom_${VERSION}_${DIM}_${WORKERS}_${METHOD}.out
#SBATCH --error=flowsom_${VERSION}_${DIM}_${WORKERS}_${METHOD}.err

# Set parameters
VERSION=${VERSION}
DIM=${DIM}
C_DIM=${C_DIM}
RAM_VALUES_PER_WORKER=${RAM_VALUES_PER_WORKER}
CHUNKSIZE=${CHUNKSIZE}
WORKERS=${WORKERS}
METHOD=${METHOD}
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_\${VERSION}_\${METHOD}_\${C_DIM}_\${DIM}_\${WORKERS}_\${CHUNKSIZE}.zarr"
IMG_LAYER="ROI1_image"
LOG_DIR=".duct/logs_flowsom_client_\${VERSION}"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${C_DIM}_\${DIM}_\${WORKERS}_\${CHUNKSIZE}"

mkdir -p "\$LOG_DIR"

# Create the dataset
pixi run --frozen -e harpy python ../benchmark_flowsom.py \\
  --dataset "\$DATASET" \\
  --c_dim "\$C_DIM" \\
  --y_dim "\$DIM" \\
  --x_dim "\$DIM" \\
  --chunksize "\$CHUNKSIZE" \\
  --img_layer "\$IMG_LAYER"

# Run the benchmark with instrumentation
pixi run --frozen -e harpy duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_flowsom.py \\
  --dataset "\$DATASET" \\
  --c_dim "\$C_DIM" \\
  --y_dim "\$DIM" \\
  --x_dim "\$DIM" \\
  --method "\$METHOD" \\
  --batches_flowsom "\$WORKERS" \\
  --threads "\$WORKERS" \\
  --workers 1 \\
  --chunksize "\$CHUNKSIZE" \\
  --img_layer "\$IMG_LAYER" \\
  --memory_limit "\$RAM_VALUES_PER_WORKER"

rm -r "\$DATASET"
EOF
  done
done
done