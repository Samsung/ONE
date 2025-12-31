# Circle Model Quantization with GGML

This tool quantizes Circle models using the GGML library.

## Prerequisites
- `gcc` installed
- `flatc` (FlatBuffers compiler) must be available
- Set `FLATC_PATH` if `flatc` is not in your PATH or standard build locations

## Building the Tool

The tool is structured as a Python package `oquantize` located in `tools/oquantize`.
It includes a C extension that needs to be compiled and generates `circle.py` from schema.

```bash
cd tools/oquantize
python3 setup.py
```

This compiles `libggml_quant.so` from the GGML source files and generates `circle.py`.

## Running the Tool

To quantize a Circle model, run the `oquantize` package from the `tools` directory:

```bash
cd tools
# Usage: python -m oquantize <quant_type> <input_circle> <output_circle>
python3 -m oquantize q4_0 prefill.circle prefill.q4.circle
python3 -m oquantize q4_0 decode.circle decode.q4.circle
```

### File Size Comparison

| File | Original Size | Quantized Size | Reduction |
|------|---------------|----------------|-----------|
| prefill.circle | 18M | 2.7M | ~85% |
| decode.circle | 18M | 2.7M | ~85% |

(Note: significant reduction is observed due to FP32 -> Q4_0 quantization).

## Implementation Details
- **Package Structure**: `tools/oquantize/`
- **C Extension**: `libggml_quant.so` compiled from `ggml-quants.c`, `ggml-aarch64.c`, and `ggml.c`
- **Quantization**: Row-wise `GGML_Q4_0` quantization for `GATHER` (input 0) and `FULLY_CONNECTED` (input 1) weights
- **Schema**: `circle.py` generated from `runtime/libs/circle-schema/circle_schema.fbs` using `flatc --python --gen-object-api --gen-onefile`
