import json
from pathlib import Path

import pandas as pd


def extend_summary(summary):
    # extract command
    command = summary['command']
    # extract any parameters with -- e.g. python test_script.py --duration 1 --memory-size=1000 to {
    #   "duration": 1,
    #   "memory-size": 1000
    # }
    params = {}
    for arg in command.split():
        if arg.startswith("--"):
            key, value = arg.split("=")
            params[key[2:]] = value
    dataset = params.get('dataset')
    if dataset is None:
        raise ValueError(f"Dataset not found in {params}")
    print(dataset)
    params['dataset_size'] = int(Path(dataset).stem.split('_')[-1])
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
    logs = list(p_logs.glob("*_info.json"))
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
    pd.DataFrame(rows).to_csv(p_logs / "summary.csv", index=False)