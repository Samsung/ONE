import argparse
import numpy as np
import psutil
import os
from typing import List
from onert import infer
# TODO: Import tensorinfo from onert
from onert.native.libnnfw_api_pybind import tensorinfo


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def parse_shapes(shape_strs: List[str]) -> List[List[int]]:
    shapes = []
    for s in shape_strs:
        try:
            shapes.append([int(dim) for dim in s.strip().split(",")])
        except Exception:
            raise ValueError(f"Invalid shape string: '{s}' (expected: 1,224,224,3 ...)")
    return shapes


def get_validated_input_tensorinfos(sess: infer.session,
                                    static_shapes: List[List[int]]) -> List[tensorinfo]:
    original_infos = sess.get_inputs_tensorinfo()
    if len(static_shapes) != len(original_infos):
        raise ValueError(
            f"Input count mismatch: model expects {len(original_infos)} inputs, but got {len(static_shapes)} shapes"
        )

    updated_infos: List[tensorinfo] = []

    for i, info in enumerate(original_infos):
        shape = static_shapes[i]
        if info.rank != len(shape):
            raise ValueError(
                f"Rank mismatch for input {i}: expected rank {info.rank}, got {len(shape)}"
            )
        info.dims = shape
        info.rank = len(shape)
        updated_infos.append(info)

    return updated_infos


def benchmark_inference(nnpackage_path: str, backends: str, input_shapes: List[List[int]],
                        repeat: int):
    mem_before_kb = get_memory_usage_mb() * 1024

    sess = infer.session(path=nnpackage_path, backends=backends)
    model_load_kb = get_memory_usage_mb() * 1024 - mem_before_kb

    input_infos = get_validated_input_tensorinfos(
        sess, input_shapes) if input_shapes else sess.get_inputs_tensorinfo()

    # Create dummy input arrays
    dummy_inputs = []
    for info in input_infos:
        shape = tuple(info.dims[:info.rank])
        dummy_inputs.append(np.random.rand(*shape).astype(info.dtype))

    prepare = total_io = total_run = 0.0

    # Warmup runs
    prepare_kb = 0
    for _ in range(3):
        outputs, metrics = sess.infer(dummy_inputs, measure=True)
        del outputs
        if "prepare_time_ms" in metrics:
            prepare = metrics["prepare_time_ms"]
            prepare_kb = get_memory_usage_mb() * 1024 - mem_before_kb

    # Benchmark runs
    for _ in range(repeat):
        outputs, metrics = sess.infer(dummy_inputs, measure=True)
        del outputs
        total_io += metrics["io_time_ms"]
        total_run += metrics["run_time_ms"]

    execute_kb = get_memory_usage_mb() * 1024 - mem_before_kb

    print("======= Inference Benchmark =======")
    print(f"- Warmup runs   : 3")
    print(f"- Measured runs : {repeat}")
    print(f"- Prepare       : {prepare:.3f} ms")
    print(f"- Avg I/O       : {total_io / repeat:.3f} ms")
    print(f"- Avg Run       : {total_run / repeat:.3f} ms")
    print("===================================")
    print("RSS")
    print(f"- MODEL_LOAD    : {model_load_kb:.0f} KB")
    print(f"- PREPARE       : {prepare_kb:.0f} KB")
    print(f"- EXECUTE       : {execute_kb:.0f} KB")
    print(f"- PEAK          : {max(model_load_kb, prepare_kb, execute_kb):.0f} KB")
    print("===================================")


def main():
    parser = argparse.ArgumentParser(description="ONERT Inference Benchmark")
    parser.add_argument("nnpackage", type=str, help="Path to .nnpackage directory")
    parser.add_argument("--backends",
                        type=str,
                        default="cpu",
                        help="Backends to use (default: cpu)")
    parser.add_argument("--input-shape",
                        nargs="+",
                        help="Input shapes for each input (e.g. 1,224,224,3 1,10)")
    parser.add_argument("--repeat",
                        type=int,
                        default=5,
                        help="Number of measured inference repetitions")

    args = parser.parse_args()
    shapes = parse_shapes(args.input_shape) if args.input_shape else None

    benchmark_inference(nnpackage_path=args.nnpackage,
                        backends=args.backends,
                        input_shapes=shapes,
                        repeat=args.repeat)


if __name__ == "__main__":
    main()
