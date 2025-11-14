#!/usr/bin/env python3
import sys
import pathlib

def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: with.py <input.circle>\\n")
        sys.exit(1)

    input_path = pathlib.Path(sys.argv[1])
    if not input_path.is_file():
        sys.stderr.write(f"File not found: {input_path}\\n")
        sys.exit(1)

    # Read the binary content of the circle file and write it to stdout
    with input_path.open('rb') as f:
        sys.stdout.buffer.write(f.read())

if __name__ == "__main__":
    main()
