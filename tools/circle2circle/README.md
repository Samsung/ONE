# circle2circle (circle to circle)

`circle2circle` is a tool for transforming Circle models.

It includes various filter command (= pass) to perform specific modifications.

<br>

## How to Use

Imagine Unix filter usage like `cat hello.txt | sort | uniq`.

All circle2circle command scripts read a Circle model from **standard input** and write the transformed model to **standard output**.

An example:

```bash
./rename.io.remove_namespace.py < in.circle > out.circle
```

Filters example:

```bash
./with.py in.circle |
    ./select.op.py --by_id 0-181 |
    ./gc.py > new.circle
```

<br>

## Filter List

### `remove.io.py`

Removes input or output tensors from a Circle model, keeping only the tensors at the specified indices.

#### Arguments

*   `io_type` (required): Specifies whether to process `input` or `output` tensors.
*   `--keep_by_name` (optional): A string defining the names of the tensors to keep. It supports comma‑separated tensor names (e.g., "input1,input2").
*   `--keep_by_id` (optional): Specifies the tensor indices to keep. Supports multiple ranges separated by commas and individual indices (e.g., "0,2-4").

**Note:** Exactly one of `--keep_by_name` or `--keep_by_id` must be provided.

##

### `rename.io.remove_namespace.py`

Removes namespaces from the names of input and output tensors. A namespace is identified as the part of the tensor name before a double colon (`::`). For example, a tensor named `module::input_tensor` would be renamed to `input_tensor`.

##

### `rename.io.remove_prefix.py`

Removes a user-specified prefix from the names of all tensors in the model.

#### Arguments

*   `prefix` (required): The string prefix to remove from tensor names.


##

### `reshape.fc_weight.py`

Reshapes the weight tensors of `FULLY_CONNECTED` operators from an effectively 2D shape (e.g., `[1, 1, D_out, D_in]`) to a strict 2D shape (`[D_out, D_in]`). This is useful for optimizing or standardizing the model structure. If a weight tensor is used by multiple operators, a new tensor is created for the specific operator to prevent conflicts.

##

### `transpose.io.kcache.py`

Finds input tensors matching the pattern `*key_cache_\d+` (e.g., `past_key_values_key_cache_0`) and transposes their second and third dimensions if they are 4D. For example, a shape `[d0, d1, d2, d3]` will become `[d0, d2, d1, d3]`.

##

### `fuse.bmm_lhs_const.py`

Fuses `BATCH_MATMUL` + `TRANSPOSE` to `FULLY_CONNECTED` when LHS is constant.

#### Transformation Diagram

```
BEFORE:

LHS(constant):[B,M,K] \
                     BatchMatMul(LHS,RHS):[B,M,N] -> TRANSPOSE:[B,N,M] -> OUTPUT
RHS:[B,K,N]         /

AFTER:

RHS:[B,K,N]         \
                     FullyConnected(RHS,LHS):[B,N,M] -> OUTPUT
LHS(constant):[B,M,K] /             ~~   ~~
                                  input weights

Condition:
- B = 1 and K = 1

Key Relationship:
- BatchMatMul's LHS (constant) becomes FullyConnected's weights
- BatchMatMul's RHS becomes FullyConnected's input
```

##

### `select.op.py`

Selectively removes operators from a Circle model based on their index range. This filter allows you to keep only the operators within specified index ranges while removing all others. It automatically handles tensor connections, updates subgraph inputs/outputs, and cleans up unused operator codes.

#### Arguments

*   `--by_id` (required): Specifies the operator index range to keep. Supports multiple ranges separated by commas and individual indices.

#### Example Usage

```bash
# Keep only operators 0-181
./select.op.py --by_id 0-181 < old.circle > new.circle

# Keep operators 0-10 and 15-20
./select.op.py --by_id 0-10,15-20 < old.circle > new.circle

# Keep only operator 5
./select.op.py --by_id 5 < old.circle > new.circle
```

##

### `gc.py`

Performs garbage collection by removing unreachable tensors and buffers, reducing model size and memory consumption.

##

### `retype.input_ids.py`

Finds tensors named `input_ids` and changes their data type from int64 to int32. This filter is useful for models that need to be compatible with hardware or frameworks that expect input_ids to be 32-bit integers instead of 64-bit integers.

##

### `gen_circle.*.py`


These scripts generate test Circle models with specific operator patterns for development and testing purposes. Each script follows the naming convention `gen_circle.<operation>.py` and automatically generates an output file with the name `<operation>.circle` when executed.

#### `gen_circle.add.py`

Generates a simple Circle model with one `ADD` operator for testing basic functionality.

#### `gen_circle.bmm_lhs_const.fc.py`

Generates a test Circle model with `BATCH_MATMUL` and `TRANSPOSE` operations where the LHS is constant. This model is designed to test the fusion pattern used in `fuse.bmm_lhs_const.py`.

## `merge.circle.py`

Merges two Circle model files into a single model by appending their subgraphs and adding signatures. The script accepts one or more Circle files (currently limited to two).

- **Positional arguments**:
  `circles` – one or more Circle model files to merge (e.g., `in1.circle in2.circle`).

- **Optional arguments**:
  `--sig-names` – semicolon‑separated signature names for the subgraphs (e.g., `"prefill;decode"`). If omitted, the script derives the signature names from the input filenames by stripping the `.circle` extension.

### Usage examples

```bash
# Merge two models, using filenames as signature names
./merge.circle.py model1.circle model2.circle

# Merge with custom signature names
./merge.circle.py model1.circle model2.circle --sig-names "prefill;decode"
```

The merged model is written to **standard output**, allowing it to be piped into other tools or redirected to a file.
