# inference_benchmark.py — Inference Performance & Memory Benchmark

This sample measures end-to-end inference latency and RSS memory usage for an ONERT model, with optional static input shapes.

## Purpose

- Load an NNFW package
- (Optionally) override input shapes
- Perform warm-up and measured inference runs
- Report:
  - Prepare / I/O / execution latency (ms)
  - RSS memory delta (KB) for model load, prepare, execute, and peak

## Usage

```bash
python inference_benchmark.py <nnpackage_path> [--backends BACKENDS] [--input-shape SHAPES …] [--repeat N]
```

| Argument           | Description                                                                      |
| ------------------ | -------------------------------------------------------------------------------- |
| `<nnpackage_path>` | Path to your `.nnpackage` directory or model file                                |
| `--backends`       | Backend to use (e.g. `cpu`, `gpu`). Default: `cpu`                                |
| `--input-shape`    | One or more comma-separated shape strings, e.g. `1,224,224,3 1,10`                 |
| `--repeat`         | Number of timed inference repetitions (after 3 warm-up runs). Default: `5`        |

## Example

```bash
# Measure on CPU with default shapes, 5 repeats
python inference_benchmark.py /path/to/model.nnpackage

# Measure on GPU with two inputs: [1,224,224,3] and [1,10], 10 repeats
python inference_benchmark.py /path/to/model.nnpackage \
  --backends gpu \
  --input-shape 1,224,224,3 1,10 \
  --repeat 10
```

## What It Does

1. Warm-up
Runs `infer()` 3 times (results discarded) to stabilize performance metrics.

2. Benchmark
Runs `infer()` N times, accumulating:
   - I/O time (`io_time_ms`)
   - Run time (`run_time_ms`)

3. Memory Measurement
Uses `psutil` to sample `RSS` before model load, after prepare, and after execution.

4. Reporting
Prints latency statistics and memory deltas:

```text
======= Inference Benchmark =======
- Warmup runs   : 3
- Measured runs : 5
- Prepare       : 12.345 ms
- Avg I/O       :  0.123 ms
- Avg Run       :  1.234 ms
===================================
RSS
- MODEL_LOAD    : 15000 KB
- PREPARE       : 30000 KB
- EXECUTE       : 32000 KB
- PEAK          : 32000 KB
===================================
```
