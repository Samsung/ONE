# TinyLlama Text Generation Example

This document provides a step‑by‑step guide for generating and processing a TinyLlama text‑generation model.

## Summary

1. Set up the environment and install dependencies.
2. Generate the initial `prefill` and `decode` Circle model files.
3. Run the pipeline to optimize, reshape, and prune the model, producing a final `decode.circle` ready for inference.

## Prerequisites

### 1. Python virtual environment
```bash
cd runtime/ggma/examples/generate_text/
python3 -m venv _
source _/bin/activate
```

### 2. Install required Python packages
```bash
pip install -r requirements.txt
```

### 3. Install TICO (Torch IR to Circle ONE)
```bash
# Clone the repository
git clone https://github.com/Samsung/TICO.git
# Install it in editable mode
pip install -e TICO
```

### 4. Get [o2o](https://github.com/Samsung/ONE/pull/16233) in PATH
*Requires the GitHub CLI (`gh`).*
```bash
gh pr checkout 16233
export PATH=../../../../tools/o2o:$PATH
```

## Generating Model Files

### 1. Create the prefill and decode Circle model files
```bash
python prefill.py   # Generates prefill.circle
python decode.py    # Generates decode_.circle
```

Verify the generated files:
```bash
ls -lh *.circle
# -rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 decode_.circle
# -rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 prefill.circle
```

### 2. Update `tinyllama.decode.circle`
Fuse attention and normalize KV-cache inputs for the decode model.

```bash
# Fuse attention and reshape KV-cache for the decode model
fuse.attention.py < decode_.circle \
    | fuse.bmm_lhs_const.py \
    | reshape.io.py input --by_shape [1,16,30,4] [1,16,32,4] \
    | transpose.io.kvcache.py > decode.circle
```

### 3. Merge prefill and decode circles
Merge the models, retype input IDs, and clean up.

```bash
merge.circles.py prefill.circle decode.circle \
    | downcast.input_ids.py \
    | gc.py > model.circle
```

Verify final model files:
```bash
ls -l {decode,prefill,model}.circle
# -rw-rw-r-- 1 gyu gyu 18594868 Nov 22 17:26 decode.circle
# -rw-rw-r-- 1 gyu gyu 18642052 Nov 22 07:53 prefill.circle
# -rw-rw-r-- 1 gyu gyu 18629520 Nov 22 17:28 model.circle
```

## Create a GGMA package

1. Create the package root directory and move `model.circle` there:
```bash
cd runtime/ggma/examples/generate_text
mkdir tinyllama
mv model.circle tinyllama/
```

2. Copy the tokenizer files (replace `{your_snapshot}` with the actual snapshot hash):
```bash
cp -L ~/.cache/huggingface/hub/models--Maykeye--TinyLLama-v0/snapshots/{your_snapshot}/tokenizer.* tinyllama/
cp -L ~/.cache/huggingface/hub/models--Maykeye--TinyLLama-v0/snapshots/{your_snapshot}/config.json tinyllama/
```

```bash
tree tinyllama/
tinyllama/
├── model.circle
├── tokenizer.json
└── tokenizer.model
```

## Build and run `ggma_run`

```bash
make -j$(nproc)
make install
```

Check version:
```bash
Product/out/bin/ggma_run --version
# ggma_run v0.1.0 (nnfw runtime: v1.31.0)
```

Run the model:
```bash
Product/out/bin/ggma_run tinyllama
# prompt: Lily picked up a flower.
# generated: { 1100, 7899, 289, 826, 351, 600, 2439, 288, 266, 3653, 31843, 1100, 7899, 289, 1261, 291, 5869, 291, 1261, 31843, 1100, 7899 }
# detokenized: She liked to play with her friends in the park. She liked to run and jump and run. She liked
```
