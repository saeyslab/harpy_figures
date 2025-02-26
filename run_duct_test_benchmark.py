import subprocess

for duration in [0.5]:
    for memory_size in [1000]:
        for method in [0, 1000, 2000, 3000, 4000, 5000]:
            # TODO: parsing requires presence of '=' in 'parameter=value'
            command = f"pixi run --frozen 'duct --sample-interval 0.1 --report-interval 0.2 python test_script.py --duration={duration} --memory-size={memory_size} --method={method}'"
            subprocess.run(command, shell=True)