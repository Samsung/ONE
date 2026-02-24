#!/usr/bin/env python3
import argparse
import shutil
import json
import yaml
import os
import sys
from config import O2O_DIR, BUILD_DIR
from util import run_cmd


def main():
    parser = argparse.ArgumentParser(description="Export model to GGMA package")
    parser.add_argument("-s",
                        "--script",
                        required=True,
                        help="Export script to use (e.g., tinyllama.py)")
    parser.add_argument("-p",
                        "--pipeline",
                        default="pipeline.yaml",
                        help="Pipeline configuration file (default: pipeline.yaml)")
    args = parser.parse_args()

    export_script_name = args.script
    pipeline_file = args.pipeline

    # Change to build directory
    if not os.path.exists(BUILD_DIR):
        print(f"Error: {BUILD_DIR} directory does not exist. Run 'opm import' first.")
        return

    os.chdir(BUILD_DIR)

    # Load pipeline configuration
    pipeline_path = os.path.abspath(pipeline_file)
    if not os.path.exists(pipeline_path):
        print(f"Error: Pipeline file {pipeline_path} not found.")
        return

    with open(pipeline_path, "r") as f:
        pipeline_config = yaml.safe_load(f)

    # Find model directory by config.json
    model_dir = None
    for d in os.listdir("."):
        config_path = os.path.join(d, "config.json")
        if os.path.isdir(d) and os.path.exists(config_path):
            model_dir = d
            print(f"Using local model directory: {model_dir}")
            break

    if not model_dir:
        raise ValueError("No local model directory found (directory with config.json)")

    # Add o2o tools and the current directory (where pipeline scripts reside) to PATH
    env = os.environ.copy()
    cwd_path = os.path.abspath(os.getcwd())
    o2o_path = os.path.abspath(os.path.join(cwd_path, "..", O2O_DIR))
    env["PATH"] = f"{o2o_path}:{cwd_path}:{env['PATH']}"
    os.environ.update(env)

    # Use current python executable instead of finding venv python
    python_bin = sys.executable
    export_script = os.path.join("..", export_script_name)

    # 1. Generate prefill and decode circles
    print(f"Running {export_script_name} (prefill)...")
    run_cmd(f"{python_bin} {export_script} --mode prefill", env=env)

    print(f"Running {export_script_name} (decode)...")
    run_cmd(f"{python_bin} {export_script} --mode decode", env=env)

    # Helper to run pipeline command
    def run_pipeline_step(step_name):
        if step_name in pipeline_config:
            print(f"Running {step_name} pipeline...")
            cmd = pipeline_config[step_name]
            # If cmd is a multiline string (from YAML |), it might contain newlines.
            # We can replace newlines with spaces or let the shell handle it if it's a single command string.
            # For safety with pipes, we replace newlines with spaces if they are meant to be a single line command.
            # But YAML block scalar preserves newlines.
            # If the user wrote it with pipes at the start of lines, we should join them.
            cmd = cmd.replace("\n", " ")
            run_cmd(cmd, env=env)

    # 2. Pipeline (decode)
    run_pipeline_step("decode")

    # 3. Merge
    run_pipeline_step("merge")

    # 4. Create package directory and copy files
    # Find source directory with tokenizer.json
    source_dir = None
    for d in os.listdir("."):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "tokenizer.json")):
            source_dir = d
            break

    if source_dir:
        package_dir = "out"
        print(f"Creating package directory {package_dir}...")
        os.makedirs(package_dir, exist_ok=True)

        # Copy tokenizer and config files
        for filename in ["tokenizer.json", "tokenizer.model", "config.json"]:
            src = os.path.join(source_dir, filename)
            if os.path.exists(src):
                shutil.copy2(src, package_dir)

        # Move model.circle
        print(f"Moving model.circle to {package_dir}...")
        shutil.move("model.circle", os.path.join(package_dir, "model.circle"))
    else:
        print(
            "Warning: Could not find source directory (directory with tokenizer.json). Leaving model.circle in current dir."
        )


if __name__ == "__main__":
    main()
