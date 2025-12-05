#!/usr/bin/env python3
import shutil
import os

import argparse
from config import VENV_DIR, O2O_DIR, BUILD_DIR


def remove_dir(path):
    """Remove directory if it exists, print message only if removed."""
    try:
        shutil.rmtree(path)
        print(f"Removing {path} directory...")
    except FileNotFoundError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Clean build artifacts")
    parser.add_argument("--all",
                        action="store_true",
                        help="Remove all generated files including venv and o2o")
    args = parser.parse_args()

    # Always remove build directory
    remove_dir(BUILD_DIR)

    if args.all:
        remove_dir(O2O_DIR)
        remove_dir(VENV_DIR)
        print("Full clean complete.")
    else:
        print("Clean complete.")


if __name__ == "__main__":
    main()
