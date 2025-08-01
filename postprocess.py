import json
from pathlib import Path

import pandas as pd


def extend_summary(summary):
    # extract command
    command = summary['command']
    print(command)
    # extract any parameters with -- e.g. python test_script.py --duration 1 --memory-size=1000 to {
    #   "duration": 1,
    #   "memory-size": 1000
    # }
    # other case: "python ../benchmark_aggregation.py --dataset /data/gent/vo/001/gvo00163/vsc40523/VIB/DATA/benchmark_harpy/sdata_50000_rechunked_4096.zarr --img_layer image_tiled --labels_layer labels_cells_harpy --threads 8 --workers 1 --method harpy"

    # this does not work for the above case
    # params = {}
    # for arg in command.split():
    #     if arg.startswith("--"):
    #         key, value = arg.split("=")
    #         params[key[2:]] = value

    # this works for the above case
    params = {}
    for i in range(len(command.split())):
        if command.split()[i].startswith("--"):
            key = command.split()[i][2:]
            if i + 1 < len(command.split()):
                value = command.split()[i + 1]
                if value.startswith("--"):
                    value = None
                    continue
                try:
                    # try to convert to int
                    value = int(value)
                except ValueError:
                    # if it fails, keep it as string
                    pass
            else:
                value = None
            params[key] = value
    dataset = params.get('dataset')
    if dataset is None:
        raise ValueError(f"Dataset not found in {params}")
    print(dataset)
    if not 'c_dim' in params:
        params['c_dim'] = 20
    # if x_dim or y_dim is not in params, get the from dataset by parsing the second to last number of the stem
    if not ('x_dim' in params or 'y_dim' in params):
        # get the second to last number of the stem
        # e.g. sdata_50000_rechunked_4096.zarr -> 4096
        # e.g. sdata_50000_rechunked_4096_x512_y512.zarr -> 512
        # e.g. sdata_50000_rechunked_4096_x512_y512_z256.zarr -> 256
        # e.g. sdata_50000_rechunked_4096_x512_y512_z256_c20.zarr -> 20
        # e.g. sdata_50000_rechunked_4096_x512_y512_z256_c20_t1.zarr -> 1
        params['x_dim'] = int(Path(dataset).stem.split('_')[-2])
        params['y_dim'] = int(Path(dataset).stem.split('_')[-2])
    params['pixels'] = params['c_dim'] * params['x_dim'] * params['y_dim']
    # params['dataset_size'] = int(Path(dataset).stem.split('_')[2])
    print(params)
    return {**summary, **params}


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", help="Path to logs", default=".duct/logs")
    args = parser.parse_args()
    p_logs = Path(args.logs).resolve()
    assert p_logs.exists(), f"Logs {p_logs} does not exist"

    # gather all _info logs
    logs = list(p_logs.glob("*info.json"))
    # read json logs and extract print all keys
    rows = []
    for log in logs:
        print(f"Processing {log}")
        if log.read_text() == "":
            print(f"Skipping {log}")
            continue
        with open(log, "r") as f:
            data = json.load(f)
            summary = data['execution_summary']
            extended_summary = extend_summary(summary)
            print(extended_summary)
            rows.append(extended_summary)
    assert len(rows) > 0, "No logs found"
    pd.DataFrame(rows).to_csv(p_logs / "summary.csv", index=False)