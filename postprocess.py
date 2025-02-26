import json

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
            params[f'config_{key[2:]}'] = value
    print(params)
    return {**summary, **params}


if __name__ == "__main__":
    import argparse
    import importlib.util
    import sys
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
        with open(log, "r") as f:
            data = json.load(f)
            summary = data['execution_summary']
            extended_summary = extend_summary(summary)
            print(extended_summary)
            rows.append(extended_summary)
    pd.DataFrame(rows).to_csv(p_logs / "summary.csv", index=False)