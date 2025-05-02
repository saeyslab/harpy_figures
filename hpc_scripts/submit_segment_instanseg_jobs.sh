#!/bin/bash
DIM_VALUES=(2000 10000 20000)
RAM_VALUES=(20 50 100)  # Memory in GB
TIME_VALUES=(1 3 10)

WORKERS_VALUES=(1 2 4 8)

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
#SBATCH --job-name=instanseg_${DIM}_${WORKERS}
#SBATCH --output=instanseg_${DIM}_${WORKERS}.out
#SBATCH --error=instanseg_${DIM}_${WORKERS}.err

# Set parameters
DIM=${DIM}
C_DIM=10
CHUNKSIZE=1000
THREADS=1
WORKERS=${WORKERS}
METHOD="instanseg"
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_\${METHOD}_\${DIM}_\${WORKERS}.zarr"
IMG_LAYER="image_tiled"
LOG_DIR=".duct/logs_segment_instanseg"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${DIM}_\${WORKERS}"

mkdir -p "\$LOG_DIR"

pixi run --frozen -e harpy duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_segmentation.py \\
  --dataset "\$DATASET" \\
  --c_dim "\$C_DIM" \\
  --y_dim "\$DIM" \\
  --x_dim "\$DIM" \\
  --chunksize "\$CHUNKSIZE" \\
  --img_layer "\$IMG_LAYER" \\
  --method "\$METHOD"
EOF
  done
done
