# dynamic_shape_inference.py — Dynamic Shape Inference Example

Demonstrates how to run inference on a model with dynamic input dimensions (`-1`) by supplying random concrete shapes on-the-fly.

## Purpose
- Load an `.nnpackage` model
- Query the model’s input tensorinfo (which may include `-1` for dynamic dims)
- Perform 10 successive inference calls, each time replacing any `-1` dimension with a random integer in [1, 10]
- Allocate NumPy input arrays matching the randomized shapes
- Run `session.infer(...)` and report progress

## Key Points

- **Dynamic dims** (`-1`) are resolved at each call by sampling a new size
- No explicit call to `update_inputs_tensorinfo()` is required—shapes are applied directly at runtime
- Useful for testing models that accept variable batch sizes or spatial dimensions

## Usage

```bash
python3 dynamic_shape_inference.py /path/to/your_model.nnpackage [backends]
```
- `/path/to/your_model.nnpackage` — path to the NNFW package or model file
- `backends` (optional) — backend string (e.g. "cpu", "gpu"); defaults to "cpu"
