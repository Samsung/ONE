# Packaging and Manifest

## Revision History

Version Major.Minor.Patch

1.0.0: Initial version
1.1.0: `configs` is added
1.2.0: `tvn` is added for supported model-types.
1.3.0: `pkg-inputs`, `pkg-outputs` and `model-connect` are added
1.3.1: `model-types` is not mandatory any longer

## 1. Overview

`nnpackage` is the input of nnfw, and the output of nncc.

`nnpackage` contains all data (such as model, `MANIFEST`, custom_op) that requires to run a given model.

The document will cover packaging and `MANIFEST` only.

For `model` and `custom_op`, see [20_model_and_operators.md](20_model_and_operators.md) and [30_custom_op.md](30_custom_op.md).

## 2. Packaging Structure

`nnpackage` is a Zip archive in the following structure:

```
nnpackage
├── custom_op
├── metadata
│   ├── MANIFEST
│   └── config.cfg
└── mymodel.model
```

- `mymodel.model` is a model file that has computation graph and weights.
- `config.cfg` is a configuration file that has parameters to configure onert.
- `metadata` is a directory that contains all metadata including `MANIFEST`.
- `MANIFEST` is a collection of attributes about this package.
- `custom_op` is a directory that contains implementation objects.

## 3. Packaging Format

`nnpackage` is contained in `Zip Archive`, which could be either `compressed` or `stored` (no compression).

## 4. Manifest

`MANIFEST` is a collection of attributes about `nnpacakge`. `MANIFEST` should be a valid JSON.

### Attributes

#### version

`version` is composed of 3 numbers in `MAJOR`.`MINOR`.`PATCH`.

Given a version number MAJOR.MINOR.PATCH, increment the:

MAJOR version when you make incompatible/breaking changes,
MINOR version when you add functionality in a backwards-compatible manner, and
PATCH version when you make backwards-compatible bug fixes.

For detail, see [semantic versioning 2.0.0](https://semver.org/)

##### major-version

`major-version` is the major version of `nnpackage`.

##### minor-version

`minor-version` is the minor version of `nnpackage`.

##### patch-version

`patch-version` is the patch version of `nnpackage`.

#### configs

`configs` is an array of configuration file names placed in `metadata` folder. This can be empty or
attribute itself can be omitted. As of now we only support only one item.

#### models

`models` is an array of path to model files, which is relative path from top level directory of this package.
The first element from the array will be the default model to be executed.

#### Multiple Models Description

`nnpackage` can describe a workflow that involves several models.
Each connection point is identified by a **triple**:

```
model_index : subgraph_index : io_index
```

* **model_index** – Position of the model in the `models` array (starting at 0).
* **subgraph_index** – Sub‑graph identifier inside the model file (used by Circle and TensorFlow Lite).
* **io_index** – Input (or output) slot number within that sub‑graph.

##### pkg‑inputs

`pkg‑inputs` is an array of strings that specifies the entry points of the whole package.
Each entry uses the triple notation described above.

```json
"pkg-inputs": [
  "0:0:0",
  "1:0:2"
]
```

##### pkg‑outputs

`pkg‑outputs` is an array of strings that lists the exit points of the package, also using the triple notation.

```json
"pkg-outputs": [
  "2:0:0",
  "2:0:1"
]
```

##### model‑connect

`model‑connect` defines how data flows between the models.
Each object contains a **source** (`from`) and one or more **destinations** (`to`), all expressed with the triple notation.

```json
"model-connect": [
  { "from": "0:0:0", "to": [ "1:0:0", "1:0:1" ] },
  { "from": "1:0:2", "to": [ "2:0:0" ] }
]
```

* The first entry connects the output `0:0:0` of model 0 to two inputs of model 1.
* The second entry routes the output `1:0:2` of model 1 to the input `2:0:0` of model 2.

These fields enable **nnpackage** to orchestrate multiple models as a single, unified graph, making it possible to build complex pipelines where the output of one model becomes the input of another.

### Example

Here is an example of `MANIFEST` with multi-model connections:

```json
{
    "major-version" : "1",
    "minor-version" : "3",
    "patch-version" : "1",
    "configs"     : [  ],
    "models"      : [
      "encoder.circle",
      "decoder.circle"
    ],
  "pkg-inputs": [
    "0:0:0"
  ],
  "pkg-outputs": [
    "1:0:0"
  ],
  "model-connect": [
    { "from": "0:0:0", "to": [ "1:0:0" ] }
  ]
}
```

This diagram illustrates the above configuration.

```
     Package Input 0

          │
          │ (0:0:0)
          │
          ▼  nnpackage
     ┌──────────────────────┐
     │                      │
     │          Model 0     │
     │  ┌─────────────────┐ │
     │  │ encoder.circle  │ │
     │  │ (subgraph 0)    │ │
     │  └─────────────────┘ │
     │        │             │
     │        │  output     │
     │        │  (0:0:0)    │
     │        │             │
     │        ▼  Model 1    │
     │  ┌─────────────────┐ │
     │  │ decoder.circle  │ │
     │  │ (subgraph 0)    │ │
     │  └─────────────────┘ │
     │         │            │
     └─────────┼────────────┘
               │ (1:0:0)
               ▼
          Package Output 0
```

## 5. Configuration file

Configuration file is a human readable plain text file having one `key=value` in each line.
- `#` is used as comment and will be ignored afterwards.
- all leading and trailing white spaces will be ignored in both `key` and `value`.

For example
```
BACKENDS=cpu
# leading/trailing space is ignored
 EXCUTOR=Linear # some comment
```

Refer `runtime/onert/core/include/util/Config.lst` file for more information of `key`.
