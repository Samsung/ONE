# minimal.py — Basic Inference Example

Demonstrates the simplest Python-API workflow to load an NNFW model, allocate inputs, run inference, and report success.

## Purpose

- Load an `.nnpackage` model
- Automatically query input tensor shapes
- Allocate zero-filled NumPy arrays for each input
- Perform a single inference call via `session.infer(...)`
- Print a confirmation message

## Usage

```bash
python minimal.py /path/to/your_model.nnpackage [backends]
```
- `/path/to/your_model.nnpackage` – path to your NNFW package or model file
- `backends` (optional) – backend string (e.g. "cpu", "gpu"); defaults to "cpu"
