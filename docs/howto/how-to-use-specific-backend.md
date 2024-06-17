# How to Use Specific Backend during Inference

ONE runtime has many ways to use specific backend during inference

## Using NNFW API

### [nnfw_set_available_backends](https://github.com/Samsung/ONE/blob/c46ddc04abdb58323fbd38389e6927f003bfaea1/runtime/onert/api/include/nnfw.h#L458)
- Multiple backends can be set and they must be separated by a semicolon (ex: "acl_cl;cpu").
- For each backend string, `libbackend_{backend}.so` will be dynamically loaded during nnfw_prepare.
- Among the multiple backends, the 1st element is used as the default backend.

## Using Environment Variable

### 1. BACKENDS
- Same as `nnfw_set_available_backends`
- Example
```bash
BACKENDS=cpu ./Product/out/bin/onert_run ...
```

### 2. OP_BACKEND_[OP_TYPE]
- Set backend for specific operator type
- Example
  - Execute `Conv2D` operator on ruy backend and others on cpu backend
```bash
OP_BACKEND_Conv2D=ruy BACKENDS="cpu;ruy" ./Product/out/bin/onert_run ...
```

### 3. OP_BACKEND_MAP
- Set backend for specific operator by its index
- Format : `<op_id>=<backend>;<op_id>=<backend>...`
- Example
  - Execute `operator 10` on `acl_cl` backend and others on `acl_neon` backend
```bash
OP_BACKEND_MAP="10=acl_cl" BACKENDS="acl_neon;acl_cl" ./Product/out/bin/onert_run ...
```
