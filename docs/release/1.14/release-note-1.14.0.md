# Release Note 1.14.0

## ONE Compiler

### Compiler Frontend

- `one-codegen` interface now distinguishes own arguments from backend's.
- Adds `RemoveUnnecessaryStridedSlice` optimization pass.
- Introduces experimental support for generating profile data.
  - Adds `--generate_profile_data` option to `one-optimize`, `one-quantize`.
