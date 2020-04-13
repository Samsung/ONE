# Requirements (or Checkpoint)

## Packaging

### Packaging Format

- [ ] PF1. support royalty free compression
- [ ] PF2. compatible with low end devices

### Manifest

- [ ] MF1. human readable
- [ ] MF2. easy to parse for several types of configuration variables.
- [ ] MF3. small binary size for parsing (since the parser will be part of runtime)

## Model

- [ ] MD1. support multiple tensor layout (such as NHWC, NCHW, etc)
  - define layout for model / submodel / other unit?
  - use operator (such as loco)
- [ ] MD2. describe operand?
  - include in operator vs. independent field for operand
  - support unspecified dimension value & unspecified rank?
- [ ] MD3. describe operation type
  - string vs. enum value?
- [ ] MD4. support many quantization
  - howto (ex. union type quantization parameter field, field handle quantization parameter table for quantization methodology)
- [ ] MD5. backward-compatibility and maintainability
