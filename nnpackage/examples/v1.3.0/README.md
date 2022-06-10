### part_add_sqrt

Use `circle-partition` for model patitioning

#### Input

Input model
- `part_add_sqrt_000.circle`

Input partition information:
- `part_add_sqrt_000.part`

```ini
[partition]
backends=cpu,acl_cl
default=cpu
comply=opcode

[OPCODE]
SQRT=acl_cl
```

#### Output

Output models
- `part_add_sqrt_000.00001_cpu.circle`
- `part_add_sqrt_000.00002_acl_cl.circle`
- `part_add_sqrt_000.00003_acl_cl.circle`

Output connection information:
- `part_add_sqrt_000.conn.json`

```json
{
  "source" : {
    "file" : "part_add_sqrt_000.circle",
    "inputs" : [ "ifm1", "ifm2" ],
    "outputs" : [ "ofm1", "ofm2" ]
  },
  "parts" : [
    {
      "file" : "part_add_sqrt_000.00001_cpu.circle",
      "inputs" : [ "ifm1", "ifm2" ],
      "outputs" : [ "add" ]
    },
    {
      "file" : "part_add_sqrt_000.00002_acl_cl.circle",
      "inputs" : [ "add" ],
      "outputs" : [ "ofm1" ]
    },
    {
      "file" : "part_add_sqrt_000.00003_acl_cl.circle",
      "inputs" : [ "add" ],
      "outputs" : [ "ofm2" ]
    }
  ]
}
```
