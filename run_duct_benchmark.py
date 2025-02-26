import subprocess

for dataset in ['dataset_0', 'dataset_1', 'dataset_2']:
    # TODO: parsing requires presence of '=' in 'parameter=value'
    command = f"pixi run --frozen -e all 'duct --sample-interval 0.5 --report-interval 1 python benchmark.py --dataset=data/{dataset}.zarr'"
    subprocess.run(command, shell=True)