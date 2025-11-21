# circle2circle (circle to circle)

`circle2circle` is a tool for transforming Circle models.

It includes various filter command (= pass) to perform specific modifications.

<br>

## How to Use

Imagine Unix filter usage like `cat hello.txt | sort | uniq`.

All circle2circle command scripts read a Circle model from **standard input** and write the transformed model to **standard output**.

An example:

Filters example:

```bash
./select.op.py --by_id 0-181 < in.circle |
./gc.py > out.circle
```

<br>

## Filter List

### `fuse.bmm_lhs_const.py`

Fuses `BATCH_MATMUL` + `TRANSPOSE` to `FULLY_CONNECTED` when LHS is constant, and automatically reshapes the weight tensors of the **newly created** `FULLY_CONNECTED` operators from effectively 2D shapes (e.g., `[1, 1, D_out, D_in]`) to strict 2D shapes (`[D_out, D_in]`).

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

#### Additional Processing

After creating each fused `FULLY_CONNECTED` operator, this script automatically reshapes its weight tensor:
- Converts effectively 2D shapes (e.g., `[1, 1, D_out, D_in]`) to strict 2D shapes (`[D_out, D_in]`)
- If a weight tensor is used by multiple operators, creates a new tensor for the specific operator to prevent conflicts
- Sets `keepNumDims = True` to preserve batch dimensions

##

### `fuse.attention.py`

Fuses attention blocks into a single `ATTENTION` operator. The script automatically detects attention blocks by searching for `FULLY_CONNECTED` operators with `attn_q_proj` weight names, making it robust to models with different prefix operators.

#### Usage

**Normal mode** (fuse attention blocks):
```bash
./fuse.attention.py < input.circle > output.circle
```

**Debug mode** (inspect operators and extract patterns):
```bash
./fuse.attention.py --debug < input.circle
```

Example output:
```
Index OpCode                           BuiltinCode   Weight Name
-----------------------------------------------------------------------------------------------
0     GATHER                               36        
...
20    FULLY_CONNECTED                       9        tico::p_model_layers_0_self_attn_q_proj_weight
21    RESHAPE                              22        
...
64    FULLY_CONNECTED                       9        tico::p_model_layers_0_self_attn_o_proj_weight
...

Searching for attention block pattern based on weight names...
Found start_op at 20 (Weight: tico::p_model_layers_0_self_attn_q_proj_weight)
Found end_op at 64 (Weight: tico::p_model_layers_0_self_attn_o_proj_weight)

Extracted range: 20 - 64
ATTENTION_PATTERN_CODES = [9, 22, 39, 9, 22, 39, 9, 22, 39, 22, 22, 18, 45, 45, 59, 2, 18, 0, 18, 45, 45, 59, 2, 18, 0, 2, 2, 22, 45, 18, 39, 18, 22, 22, 126, 22, 0, 25, 22, 22, 126, 22, 39, 22, 9]
Verified: Pattern starts with FULLY_CONNECTED
Verified: Pattern ends with FULLY_CONNECTED
```

#### Features

- **Dynamic detection**: Automatically finds attention block start offset by searching for weight names
- **Pattern-based fusion**: Fuses 45-operator attention blocks (stride of 65 operators between blocks)
- **Debug mode**: Provides operator inspection and pattern extraction for analysis



### `transpose.io.kcache.py`

Finds input tensors matching the pattern `*key_cache_\d+` (e.g., `past_key_values_key_cache_0`) and transposes their second and third dimensions if they are 4D. For example, a shape `[d0, d1, d2, d3]` will become `[d0, d2, d1, d3]`.

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

## `merge.circles.py`

Merges multiple Circle model files into a single model by appending their subgraphs and adding signatures. The script accepts any number of Circle files.

- **Positional arguments**:
  `circles` – one or more Circle model files to merge (e.g., `in1.circle in2.circle in3.circle ...`).

- **Optional arguments**:
  `--sig-names` – semicolon‑separated signature names for the subgraphs (e.g., `"prefill;decode;extra"`). If omitted, the script derives the signature names from the input filenames by stripping the `.circle` extension.

### Features

- **N-model merging**: Supports merging any number of input models (not limited to two).
- **Operator code deduplication**: Identical operator codes are merged to reduce redundancy.
- **Buffer deduplication**: Buffers with identical content (e.g., shared weights) are automatically deduplicated using SHA256 hashing, reducing the merged model size.

### Usage examples

```bash
# Merge multiple models (N models)
./merge.circles.py prefill.circle decode.circle > merged.circle

# Merge three models with custom signature names
./merge.circles.py model1.circle model2.circle model3.circle --sig-names "prefill;decode;extra"
```

The merged model is written to **standard output**, allowing it to be piped into other tools or redirected to a file.
