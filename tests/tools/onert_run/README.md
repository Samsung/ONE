# onert_run

`onert_run` is a tool to run `nnpackage`.

It takes `nnpackage` as input. It uses **runtime API** internally.

## Usage

### Simple run

This will run with random input data

```
$ ./onert_run path_to_nnpackage_directory
```

Output would look like:

```
nnfw_prepare takes 425.235 ms
nnfw_run     takes 2.525 ms
```
