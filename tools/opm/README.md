# OPM (ONE or circle Package Manager)

`opm` is a set of utility scripts to simplify the process of creating GGMA packages for ONE runtime. It handles environment setup, model downloading, and the export pipeline.

## Usage

The tools are designed to be run via the `opm` wrapper script (or directly via python if preferred).

### 1. Init

Initialize the environment. This creates a virtual environment, installs dependencies (including a CPU-only version of torch to avoid large downloads), and fetches the necessary `o2o` tools from the ONE repository.

```bash
$ opm init
```

### 2. Import

Download a model.

```bash
$ opm import <model_id|url> [-r <requirements_file>]
```

- `<model_id|url>`: HuggingFace model ID (e.g., `Maykeye/TinyLLama-v0`) or direct URL.
- `-r, --requirements`: (Optional) Path to a requirements file to install specific dependencies for the model.

Example:
```bash
$ opm import Maykeye/TinyLLama-v0 -r tinyllama/tinyllama.requirements
```

### 3. Export

Export the downloaded model to a GGMA package (`.circle` file + tokenizer). This runs the specified export script and pipeline configuration.

```bash
$ opm export -s <export_script> -p <pipeline_config>
```

- `-s, --script`: Path to the python export script (e.g., `tinyllama/tinyllama.py`).
- `-p, --pipeline`: Path to the pipeline configuration YAML file (e.g., `tinyllama/tinyllama.pipeline`).

Example:
```bash
$ opm export -s tinyllama/tinyllama.py -p tinyllama/tinyllama.pipeline
```

### 4. Clean

Clean up build artifacts.

```bash
$ opm clean [--all]
```

- Default: Removes the `build/` directory.
- `--all`: Removes `build/`, `venv/`, `o2o/`, and `TICO/` (full reset).
