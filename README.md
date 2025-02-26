# Figures for the Harpy manuscript

## Install
This benchmark is Linux only. Install [Pixi](https://pixi.sh/latest/).

```
pixi i -e all
```

## Usage

### Prep datasets
```bash
pixi run prep_datasets
```

### Run time benchmark
A way to run the benchmark with Hyperfine, but this has no support for memory benchmarking.

```bash
pixi run hyperfine -M 3 -L dataset dataset_0,dataset_1 -n dataset_0 -n dataset_1 -w 1 'pixi run -e all --frozen python benchmark.py data/{dataset}.zarr'
```

### Run time and memory benchmark
```bash
rm -r .duct # remove previous logs
pixi run duct_benchmark # create duct logs by calling duct on each benchmark sample
pixi run python postprocess.py # creates a summary.csv from the duct logs
```

Run the `figures.ipynb` notebook with the output .csv in `.duct/logs` to create figures.

# TODO

- [ ] create one benchmark output figure with chosen methods, parameters, duct logs and figures.
- [ ] add more datasets
- [ ] add more methods
- [ ] add support for N repetitions and error bars in figures
