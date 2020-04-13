# Packaging and Manifest

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
│   └── MANIFEST
└── mymodel.model
```

- `mymodel.model` is a model file that has computation graph and weights.
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

#### models

`models` is an array of path to model files, which is relative path from top level directory of this package.
The first element from the array will be the default model to be executed.

#### model-types

`model-types` is an array of strings that describes the type of each model in `models`.

It can have the values (case-sensitive) in following table.

| name   | description            |
|--------|------------------------|
| tflite | tensorflow lite schema |
| circle | nnpackage schema       |

### Example

Here is an example of `MANIFEST`.

```
{
    "major-version" : "1",
    "minor-version" : "0",
    "patch-version" : "0",
    "models"      : [ "mymodel.model", "yourmodel.model" ],
    "model-types" : [ "tflite", "circle" ]
}
```
