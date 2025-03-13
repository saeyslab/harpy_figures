#!/bin/bash
# Define your parameter arrays:
DIM_VALUES=(2000 10000 20000 50000)
RAM_VALUES=(10 30 30 40)  # Memory in GB
TIME_VALUES=(01 01 01 02)

WORKERS_VALUES=(1 2 4 8)

METHOD="harpy"

for i in "${!DIM_VALUES[@]}"; do
  DIM=${DIM_VALUES[$i]}
  MEM=${RAM_VALUES[$i]}
  TIME=${TIME_VALUES[$i]}
  for WORKERS in "${WORKERS_VALUES[@]}"; do
    sbatch <<EOF
#!/bin/bash
#SBATCH --partition=doduo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${WORKERS}
#SBATCH --time=${TIME}:00:00
#SBATCH --mem=${MEM}G
#SBATCH --job-name=aggregation_${METHOD}_${DIM}_${WORKERS}
#SBATCH --output=aggregation_${METHOD}_${DIM}_${WORKERS}.out
#SBATCH --error=aggregation_${METHOD}_${DIM}_${WORKERS}.err

# Set parameters
DIM=${DIM}
C_DIM=10
THREADS=${WORKERS}
WORKERS=1
METHOD=${METHOD}
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_\${DIM}_rechunked_4096.zarr"
IMG_LAYER="image_tiled"
LABELS_LAYER="labels_cells_harpy"
LOG_DIR=".duct/logs_aggregation"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${DIM}_\${THREADS}"

mkdir -p "\$LOG_DIR"

monitor -d 5 -l \${LOG_PREFIX}monitor.log -- pixi run --frozen -e harpy duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_aggregation.py \\
  --dataset "\$DATASET" \\
  --img_layer "\$IMG_LAYER" \\
  --labels_layer  "\$LABELS_LAYER" \\
  --threads "\$THREADS" \\
  --workers 1 \\
  --method "\$METHOD"
EOF
  done
done

