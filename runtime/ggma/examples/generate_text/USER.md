# Text Generation User Guide

This guide shows how to create a GGMA package for text generation models using the `opm` (one packaging manager) tool.

We use TinyLlama as an example throughout this guide.

## Creating a GGMA package

NOTE: Start from the ONE repository root directory.

### 1. Initialize environment (one-time setup)

Add [opm](../../../../tools/opm/README.md) to PATH:
```bash
$ export PATH=$PWD/tools/opm:$PATH
```

Then, change directory to tinyllama example directory and run opm init:
```bash
$ cd runtime/ggma/examples/generate_text/tinyllama
$ opm init
```

Python environment and o2o tools are prepared:
```bash
$ ls -ld o2o venv
drwxrwxr-x 2 opm opm 4096 Nov 24 09:44 o2o
drwxrwxr-x 6 opm opm 4096 Nov 24 09:42 venv
```

> **Note**: The `o2o` directory will be removed once [#13689](https://github.com/Samsung/ONE/pull/13689) is merged.

### 2. Import model from HuggingFace

```bash
$ opm import Maykeye/TinyLLama-v0
```

The HuggingFace model is downloaded to `build/tinyllama-v0/`:
```
$ tree build
build
└── tinyllama-v0
    ├── backup
    ├── config.json
    ├── demo.py
    ├── generation_config.json
    ├── model.onnx
    ├── model.safetensors
    ├── pytorch_model.bin
    ├── README.md
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    ├── tokenizer.model
    ├── train.ipynb
    └── valid.py
```

### 3. Export to GGMA package

```bash
$ opm export -s tinyllama.py
```

The GGMA package is generated in `build/out/`:
```
$ tree build/out
build/out/
├── config.json
├── model.circle
├── tokenizer.json
└── tokenizer.model
```

## Building GGMA and Running a GGMA package

NOTE: Start from the ONE repository root directory.

### Build

```bash
$ make -j$(nproc)
$ make install
```

For detailed build instructions, see the [ONE Runtime Build Guide](https://github.com/Samsung/ONE/blob/master/docs/runtime/README.md).

Confirm that `ggma_run` is built and show its version:
```bash
$ Product/out/bin/ggma_run --version
ggma_run v0.1.0 (nnfw runtime: v1.31.0)
```

### Run

Execute the GGMA package (default prompt) to see a sample output:
```bash
$ Product/out/bin/ggma_run build/out
prompt: Lily picked up a flower.
generated: { 1100, 7899, 289, 826, 351, 600, 2439, 288, 266, 3653, 31843, 1100, 7899, 289, 1261, 291, 5869, 291, 1261, 31843, 1100, 7899 }
detokenized: She liked to play with her friends in the park. She liked to run and jump and run. She liked
```

For detailed run instructions, see the [ggma_run guide](https://github.com/Samsung/ONE/blob/master/runtime/tests/tools/ggma_run/README.md).


For developers who want to understand what happens under the hood, see [DEVELOPER.md](DEVELOPER.md).
