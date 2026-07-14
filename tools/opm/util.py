import subprocess


def run_cmd(cmd, cwd=None, env=None, check=True):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, cwd=cwd, env=env, check=check)
