# TinyLlama Example Documentation

This document provides a step‑by‑step guide for generating and processing a text generation model.

## Summary

1. Set up the environment and install dependencies.
2. Generate the initial `prefill` and `decode` Circle model files.
3. Run the pipeline to optimize, reshape, and prune the model, producing a final `decode.circle` ready for inference.

## Prerequisites

1. **Python virtual environment**
   ```bash
   cd runtime/ggma/examples/generate_text/
   python3 -m venv _
   source _/bin/activate
   ```

2. **Install required Python packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TICO (Torch IR to Circle ONE)**
   ```bash
   # Clone the repository
   git clone https://github.com/Samsung/TICO.git
   # Install it in editable mode
   pip install -e TICO
   ```

## Generating Model Files

Run the provided scripts to create the prefill and decode Circle model files:

```bash
python prefill.py   # Generates tinyllama.prefill.circle
python decode.py    # Generates tinyllama.decode.circle
```

You can verify the generated files:

```bash
ls -lh *.circle
# Expected output:
# -rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 tinyllama.decode.circle
# -rw-rw-r-- 1 gyu gyu 18M Nov 14 14:09 tinyllama.prefill.circle
```

## Full Processing Pipeline

The following pipeline shows how to chain several tools to transform the model:

```bash
with.py tinyllama.decode.circle |
fuse.attention.py \
fuse.bmm_lhs_const.py | reshape.fc_weight.py | \
reshape.io.py input --by_shape [1,16,30,4] [1,16,32,4] | \
transpose.io.kvcache.py | \
remove.io.py output --keep_by_id 0 | \
select.op.py --by_id 0-181 | \
gc.py | \
retype.input_ids.py > decode.circle
```

### Explanation of each step

| Tool | Purpose |
|------|---------|
| `with.py` | Reads the Circle model from stdin and writes it to stdout. |
| `fuse.attention.py` | Fuses attention‑related operators for optimization. |
| `fuse.bmm_lhs_const.py` | Fuses constant left‑hand side matrices in batch matrix multiplication. |
| `reshape.fc_weight.py` | Reshapes fully‑connected layer weights. |
| `reshape.io.py input --by_shape [...]` | Reshapes input tensors to the specified shapes. |
| `transpose.io.kvcache.py` | Transposes the KV‑cache tensors. |
| `remove.io.py output --keep_by_id 0` | Keeps only the output tensor with ID 0, removing the rest. |
| `select.op.py --by_id 0-181` | Selects operators with IDs from 0 to 181. |
| `gc.py` | Performs garbage collection, removing unused tensors and operators. |
| `retype.input_ids.py` | Changes the data type of input IDs as needed. |
| `> decode.circle` | Saves the final processed model to `decode.circle`. |


Feel free to adjust the pipeline arguments (e.g., shapes, IDs) to suit your specific model configuration.
