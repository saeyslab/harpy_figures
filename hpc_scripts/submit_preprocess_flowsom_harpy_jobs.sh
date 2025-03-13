#!/bin/bash
C_DIM=10
CHUNKSIZE=2048
DIM_VALUES=(20000)
RAM_VALUES=(100)  # Memory in GB
TIME_VALUES=(1)

WORKERS_VALUES=(1 2 4 8)

for i in "${!DIM_VALUES[@]}"; do
  DIM=${DIM_VALUES[$i]}
  MEM=${RAM_VALUES[$i]}
  TIME=${TIME_VALUES[$i]}
  for WORKERS in "${WORKERS_VALUES[@]}"; do
    #RAM_VALUES_PER_WORKER=$(( MEM / WORKERS ))
    RAM_VALUES_PER_WORKER=$(( MEM ))

    echo $RAM_VALUES_PER_WORKER
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=doduo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${WORKERS}
#SBATCH --time=${TIME}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --job-name=preprocess_flowsom_threads_${DIM}_${WORKERS}
#SBATCH --output=preprocess_flowsom_threads_${DIM}_${WORKERS}.out
#SBATCH --error=preprocess_flowsom_threads_${DIM}_${WORKERS}.err

# Set parameters
DIM=${DIM}
C_DIM=${C_DIM}
RAM_VALUES_PER_WORKER=${RAM_VALUES_PER_WORKER}
CHUNKSIZE=${CHUNKSIZE}
WORKERS=${WORKERS}
METHOD="preprocess_flowsom_threads"
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_\${METHOD}_\${C_DIM}_\${DIM}_\${WORKERS}_\${CHUNKSIZE}.zarr"
IMG_LAYER="image"
LOG_DIR=".duct/logs_preprocess_flowsom_threads"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${C_DIM}_\${DIM}_\${WORKERS}_\${CHUNKSIZE}"

mkdir -p "\$LOG_DIR"

monitor -d 5 -l \${LOG_PREFIX}monitor.log -- pixi run --frozen -e harpy duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_preprocess_flowsom.py \\
  --dataset "\$DATASET" \\
  --c_dim "\$C_DIM" \\
  --y_dim "\$DIM" \\
  --x_dim "\$DIM" \\
  --chunksize "\$CHUNKSIZE" \\
  --img_layer "\$IMG_LAYER" \\
  --threads "\$WORKERS" \\
  --workers 1 \\
  --memory_limit "\$RAM_VALUES_PER_WORKER"
EOF
  done
done

