import subprocess

for duration in [1, 2, 3]:
    for memory_size in [1000, 2000, 3000]:
        # TODO: parsing requires presence of '=' in 'parameter=value'
        command = f"pixi run --frozen 'duct --sample-interval 0.5 --report-interval 1 python test_script.py --duration={duration} --memory-size={memory_size}'"
        subprocess.run(command, shell=True)