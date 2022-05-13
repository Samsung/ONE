# circle-partitioner

_circle-partitioner_ provides model partitioning of circle model to two or more circle models.

## How circle-partitioner work

_circle-partitioner_ requires 3 arguments for inputs files
- `--part_file`: `partition` file, use extension `.part`
- `--input_file`: `input` circle model file
- `--work_path`: `work` path where input files reside. this is optional and CWD if omitted

And options to override `partition` file as a helper to try out without editing `partition` file.
- `--backends`: override `backends` of `[partition]` section
- `--default`: override `default` of `[partition]` section

_circle-partitoner_ will read the `partition` and `input` files and group nodes with same backend
and store them into new circle models in `work` folder, where the `partition` and `input` files
are read from `work` folder.

Outputs are (1) one or more partitioned circle models and (2) connection file that gives how
the partitioned models should be connected to act like the source `input` model.

Why does input files be placed in `work` path too?
- this is still work in progress condition
- use cases are still ambigious
- original `input` model file can be used by the backend, so `.conn` file links it as `source`
- to make things simple for the backend, it would be better not to use relative path for the files

### `partition` file

`partition` follows INI format of _crew_ project.

Several example files exist in _circle-part-value-test_ `parts` folder.

This section will explain with `Net_InstanceNorm_003.part` file as example.
```ini
[partition]
backends=cpu,acl_cl
default=cpu
comply=opcode

[OPCODE]
DIV=acl_cl
```

##### `[partition]` section

`[partition]` section is the main section first to read.
- `backends`: Existing partition group names which nodes should be placed, in CSV format.
- `default`: Default group name which should be one of `backends` item.
- `comply`: How to group nodes of the model.
   - currently `opcode` and `opname` are supported
   - future work: set group by sequence number.

##### `[OPCODE`] section

This section provides how to group nodes in OPCODE types.
Nodes with same OPCODE will be grouped to that type.
This does not mean number of output circle files will be same as number of backends.
Number of output circle files will depend also on the network structure.

For above example, all `DIV` OPCODE nodes will be grouped to `acl_cl` backend.

`[OPCODE]` can override `default` backend set from `[partition]` section by using `_`.

For example, we can change default to `cpu`.
```
[OPCODE]
_=cpu
DIV=acl_cl
```

### `circle` file

Just normal `circle` file. Currently partition is supported in limited properties and
models with these properties are not support yet;
- Have multiple subgraph models
- Operators with multiple output nodes such as IF or WHILE.

### `work` folder

`partition` and `circle` file should reside in `work` folder. Output files will be
generated inside this folder.

### Example

Typical source of paritioning
```
$ tree Net_InstanceNorm_003/
Net_InstanceNorm_003/
├── Net_InstanceNorm_003.circle
└── Net_InstanceNorm_003.part
```

Command example
```
./circle-partitioner --part_file Net_InstanceNorm_003.part \
--input_file Net_InstanceNorm_003.circle --work_path= Net_InstanceNorm_003
```

Result of _circle-partitioner_
```
$ tree Net_InstanceNorm_003/
Net_InstanceNorm_003/
├── Net_InstanceNorm_003.00001_cpu.circle
├── Net_InstanceNorm_003.00002_acl_cl.circle
├── Net_InstanceNorm_003.00003_cpu.circle
├── Net_InstanceNorm_003.circle
├── Net_InstanceNorm_003.conn.ini
├── Net_InstanceNorm_003.conn.json
└── Net_InstanceNorm_003.part
```

### `Net_InstanceNorm_003.conn.ini` and `Net_InstanceNorm_003.conn.json`

These two files are identical in content but in different formats.

`.conn` file provides an information how to reconstruct the partitioned models,
`Net_InstanceNorm_003.00001_cpu.circle`, `Net_InstanceNorm_003.00002_acl_cl.circle`
and `Net_InstanceNorm_003.00003_cpu.circle`, so that it will identical to
source `Net_InstanceNorm_003.circle` model in computational results.

Here, meaning of `reconstruct` is connection of outputs and inputs of partitioned
models.

```json
$ cat Net_InstanceNorm_003/Net_InstanceNorm_003.conn.json
{
  "source" : {
    "file" : "Net_InstanceNorm_003.circle",
    "inputs" : [ "Input" ],
    "outputs" : [ "Add_as_terminal" ]
  },
  "parts" : [
    {
      "file" : "Net_InstanceNorm_003.00001_cpu.circle",
      "inputs" : [ "Input" ],
      "outputs" : [ "Pow", "Sub" ]
    },
    {
      "file" : "Net_InstanceNorm_003.00002_acl_cl.circle",
      "inputs" : [ "Sub", "Pow" ],
      "outputs" : [ "Div" ]
    },
    {
      "file" : "Net_InstanceNorm_003.00003_cpu.circle",
      "inputs" : [ "Div" ],
      "outputs" : [ "Add_as_terminal" ]
    }
  ]
}
```
Above file is in `JSON` format with `source` file and `parts` for partitioned models.
Each `parts` have `file` for the file, `inputs` for input nodes and `outputs`
for output nodes.

From the `source` we can identify inputs and outputs for the model.

- Each items in `outputs` should connect to `inputs` of another item of `parts` model,
or should be one of the `outputs` of the `source` model.
- For first `Net_InstanceNorm_003.00001_cpu.circle` model, `inputs` is(are) same
as the `source` model: `[ "Input" ]`.
- `outputs` `[ "Pow", "Sub" ]` have same names in the second model
`Net_InstanceNorm_003.00002_acl_cl.circle` which they should be connected.
- And `outputs` `[ "Div" ]` should be connected to `inputs` of
third model `Net_InstanceNorm_003.00003_cpu.circle`.

### Execution example

Consider partitioning with backends of OneRT
- `cpu`, `acl_cl`, `acl_neon`, `ruy`, `xnnpack`

Let's try with this command:
```
circle-partitioner \
   --backends cpu,acl_cl --default cpu \
   --part_file Net_InstanceNorm_003.part \
   --input_file Net_InstanceNorm_003.circle \
   --work_path Net_InstanceNorm_003
```

where `Net_InstanceNorm_003.part` is like this for initial design
```
[partition]
backends=cpu,acl_cl
default=cpu
comply=opcode

[OPCODE]
ADD=acl_cl
```
where in `[partition]` section,
- `backends` is available backends and can be overridden by `--backends`
- `default` is default backend for OpCodes not assigned in `[OPCODE]` section can be overridden by `--default`
- `comply` is which rule to apply, where only `opcode` is available for now

#### Use Op name to assign backend

```
[OP]
Reduction_indices=GPU
```
- there are very long names that may be inconvenient

### Partitioned output

#### Output files

After partition is applied, output files will look something like these
- `Net_InstanceNorm_003.part.00001_cpu.circle`
- `Net_InstanceNorm_003.part.00002_acl_cl.circle`
- `Net_InstanceNorm_003.part.00003_cpu.circle`
- `Net_InstanceNorm_003.part.conn.ini`
- `Net_InstanceNorm_003.part.conn.json`

Assume only `Div` node is assigned to `acl_cl`

#### Connection information of partitioned circle files

##### Format with ini
- `Net_InstanceNorm_003.conn.ini` provides connection of each circle files.
```
[source]
file=Net_InstanceNorm_003.circle
i1=Input
o1=Add_as_terminal

[models]
m1=Net_InstanceNorm_003.part.00001_cpu.circle
m2=Net_InstanceNorm_003.part.00002_acl_cl.circle
m3=Net_InstanceNorm_003.part.00003_cpu.circle

[Net_InstanceNorm_003.part.00001_cpu.circle]
file=Net_InstanceNorm_003.part.00001_cpu.circle
i1=Input
o1=Pow
o2=Sub

[Net_InstanceNorm_003.part.00002_acl_cl.circle]
file=Net_InstanceNorm_003.part.00002_acl_cl.circle
i1=Sub
i2=Pow
o1=Div

[Net_InstanceNorm_003.part.00003_cpu.circle]
file=Net_InstanceNorm_003.part.00003_cpu.circle
i1=Div
o1=Add_as_terminal
```

Predefined section
- `source`: Source circle model information. Has `file` as filename, `iN` for inputs and `oN` for outputs.
- `models`: Partitioned circle models. Has `mN` for model filename.

Partitioned Model section
- `iN`: inputs of this model
- `oN`: outputs of this model

In graph diagram, output order of `Net_InstanceNorm_003.part.00001_cpu.circle`
looks like `Pow,Sub` but `Div` Op in `Net_InstanceNorm_003.part.00002_acl_cl.circle`
requires order of `Sub,Pow`.

##### Format with JSON
- Use JSON format, `Net_InstanceNorm_003.part.conn.json`
```json
{
  "source" : {
    "file" : "Net_InstanceNorm_003.circle",
    "inputs" : [ "Input" ],
    "outputs" : [ "Add_as_terminal" ]
  },
  "parts" : [
    {
      "file" : "Net_InstanceNorm_003.part.00001_cpu.circle",
      "inputs" : [ "Input" ],
      "outputs" : [ "Pow", "Sub" ],
    },
    {
      "file" : "Net_InstanceNorm_003.part.00002_acl_cl.circle",
      "inputs" : [ "Pow", "Sub" ],
      "outputs" : [ "Div" ]
    },
    {
      "file" : "Net_InstanceNorm_003.part.00003_cpu.circle",
      "inputs" : [ "Div" ],
      "outputs" : [ "Add_as_terminal" ]
    }
  ]
}
```

### Future works

How to partition with multiple inputs?
