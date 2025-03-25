#!/bin/bash
#DIM_VALUES=(2000 10000 20000)
#RAM_VALUES=(30 30 50)  # Memory in GB
#TIME_VALUES=(02 08 25)

#DIM_VALUES=(50000)
#RAM_VALUES=(150)  # Memory in GB, took 100GB for other workers values (2,4,8)
#TIME_VALUES=(50)

#WORKERS_VALUES=(4)

DIM_VALUES=(2000)
RAM_VALUES=(60)  # Memory in GB, took 100GB for other workers values (2,4,8)
TIME_VALUES=(1)

WORKERS_VALUES=(1 2 8)

#DIM_VALUES=(20000)
#RAM_VALUES=(60)  # Memory in GB, took 100GB for other workers values (2,4,8)
#TIME_VALUES=(8)

#WORKERS_VALUES=(2)

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
METHOD="sopa"
DATASET="\$VSC_DATA_VO_USER/VIB/DATA/benchmark_harpy/sdata_\${METHOD}_\${DIM}_\${WORKERS}.zarr"
IMG_LAYER="image_tiled"
LOG_DIR=".duct/logs_segment_sopa"
LOG_PREFIX="\${LOG_DIR}/\${METHOD}_\${DIM}_\${WORKERS}"

mkdir -p "\$LOG_DIR"

monitor -d 5 -l \${LOG_PREFIX}monitor.log -- pixi run --frozen -e sopa duct -p \${LOG_PREFIX} \\
  --sample-interval 0.5 \\
  --report-interval 5 \\
  python ../benchmark_segmentation.py \\
  --dataset "\$DATASET" \\
  --c_dim "\$C_DIM" \\
  --y_dim "\$DIM" \\
  --x_dim "\$DIM" \\
  --chunksize "\$CHUNKSIZE" \\
  --img_layer "\$IMG_LAYER" \\
  --threads "\$THREADS" \\
  --workers "\$WORKERS" \\
  --method "\$METHOD"
EOF
  done
done
