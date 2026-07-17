# TinyLlama Text Generation Developer Guide

This document provides a detailed technical guide for generating, processing, and optimizing the TinyLlama text-generation model. For basic usage, see [USER.md](USER.md).

## Summary

1. Set up the environment and install dependencies.
2. Generate the initial `prefill` and `decode` Circle model files.
3. Run the pipeline to optimize, reshape, and prune the model, producing a final `decode.circle` ready for inference.

## Prerequisites

### 1. Python virtual environment
```bash
$ cd runtime/ggma/examples/generate_text/
$ python3 -m venv _
$ source _/bin/activate
```

### 2. Prepare [gyu](tools/gyu/README.md) and o2o tools
Install dependencies and setup `o2o` tools (similar to what `tools/gyu/init.py` does).

> **Note**: We install the CPU version of `torch` first because `gyu` depends on `TICO`, which by default pulls in the large NVIDIA version of `torch`. Installing the CPU version beforehand prevents this.

```bash
# 1. Install torch (CPU) and gyu requirements
$ pip install torch --index-url https://download.pytorch.org/whl/cpu
$ pip install -r tools/gyu/requirements.txt

# 2. Fetch o2o tools from PR #16233
$ git fetch origin pull/16233/head:pr-16233
$ git checkout pr-16233 -- tools/o2o
$ chmod +x tools/o2o/*.py

# 3. Add tools to PATH
$ export PATH=$PWD/tools/o2o:$PWD/tools/gyu:$PATH
```



## Generating Model Files

### 1. Install model dependencies
```bash
$ pip install -r tinyllama/tinyllama.requirements
```

### 2. Create the prefill and decode Circle model files
```bash
$ python tinyllama/tinyllama.py --mode prefill   # Generates prefill.circle
$ python tinyllama/tinyllama.py --mode decode    # Generates decode_.circle
```

Verify the generated files:
```bash
$ ls -lh *.circle
-rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 decode_.circle
-rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 prefill.circle
```

### 3. Update `tinyllama.decode.circle`
Fuse attention and normalize KV-cache inputs for the decode model.

```bash
$ fuse.attention.py < decode_.circle \
    | reshape.io.py input --by_shape [1,16,30,4] [1,16,32,4] \
    | transpose.io.kvcache.py > decode.circle
```

### 4. Merge prefill and decode circles
Merge the models, retype input IDs, and clean up.

```bash
$ merge.circles.py prefill.circle decode.circle \
    | fuse.bmm_lhs_const.py \
    | downcast.input_ids.py \
    | gc.py > model.circle
```

Verify final model files:
```bash
$ ls -l {decode,prefill,model}.circle
-rw-rw-r-- 1 gyu gyu 18594868 Nov 22 17:26 decode.circle
-rw-rw-r-- 1 gyu gyu 18642052 Nov 22 07:53 prefill.circle
-rw-rw-r-- 1 gyu gyu 18629520 Nov 22 17:28 model.circle
```

## Create a GGMA package

1. Create the package root directory and move `model.circle` there:
```bash
$ cd runtime/ggma/examples/generate_text
$ mkdir tinyllama
$ mv model.circle tinyllama/
```

2. Copy the tokenizer files (replace `{your_snapshot}` with the actual snapshot hash):
```bash
$ cp -L ~/.cache/huggingface/hub/models--Maykeye--TinyLLama-v0/snapshots/{your_snapshot}/tokenizer.* tinyllama/
$ cp -L ~/.cache/huggingface/hub/models--Maykeye--TinyLLama-v0/snapshots/{your_snapshot}/config.json tinyllama/
```

```bash
$ tree tinyllama/
tinyllama/
├── model.circle
├── tokenizer.json
└── tokenizer.model
```

## Build and run `ggma_run`

```bash
$ make -j$(nproc)
$ make install
```

Check version:
```bash
$ Product/out/bin/ggma_run --version
ggma_run v0.1.0 (nnfw runtime: v1.31.0)
```

Run the model:
```bash
$ Product/out/bin/ggma_run tinyllama
prompt: Lily picked up a flower.
generated: { 1100, 7899, 289, 826, 351, 600, 2439, 288, 266, 3653, 31843, 1100, 7899, 289, 1261, 291, 5869, 291, 1261, 31843, 1100, 7899 }
detokenized: She liked to play with her friends in the park. She liked to run and jump and run. She liked
```
