import subprocess
from pathlib import Path


def run_benchmark(method, dataset, threads):
    for threads in [1, 2, 4, 8]:
        for dataset in list(p_data.glob("dataset_*.zarr")):
            if 'sopa' in method:
                environment = 'sopa'
            else:
                environment = method
            timeout = 30 # seconds
            # TODO: parsing requires presence of '=' in 'parameter=value'
            cmd_benchmark = f"timeout {timeout} duct --sample-interval 0.5 --report-interval 1 python benchmark.py --method={method} --dataset={dataset} --threads={threads}"
            command = f"pixi run --frozen -e {environment} '{cmd_benchmark}'"
            p = subprocess.Popen(command, shell=True)
            try: 
                p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                return False
            finally:
                p.terminate()
            # if p return code is not 0, then skip this method
            if p.returncode != 0:
                return False
            print(f"Finished {command}")
    return True


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    p_data = root / "data"
    assert p_data.exists(), f"Data {p_data} does not exist"
    
    for method in ['harpy', 'sopa', 'sopa_dask']:
        run_benchmark(method, p_data, 1)
    print("Finished all benchmarks")