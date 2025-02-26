# Figures for the Harpy manuscript

## Install
This benchmark is Linux only. Install [Pixi](https://pixi.sh/latest/).

```
pixi i -e all
```

## Usage

### Prep datasets
```
pixi run prep_datasets
```

### Run time benchmark
```
pixi run hyperfine -M 3 -L dataset dataset_0,dataset_1 -n dataset_0 -n dataset_1 -w 1 'pixi run -e all --frozen python benchmark.py data/{dataset}.zarr'
```

### Run memory benchmark
