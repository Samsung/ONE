# model2nnpkg

`model2nnpkg` is a tool to convert model (e.g. `tflite`, `circle` or `tvn`) to `nnpackage`.

It takes `modelfile` as input and generates `nnpackage`.

## Prerequisite

Install jq
```
$ sudo apt-get install jq
```

## Usage

```
Convert modelfile (tflite, circle or tvn) to nnpackage.

Options:
    -h   show this help
    -o   set nnpackage output directory (default=.)
    -p   set nnpackage output name (default=[1st modelfile name])
    -c   provide configuration files
    -m   provide model files
    -i   provide files of models' information for adding connection information between models in MANIFEST
         This option is for multi-model. You don't need to use this option if you want to create nnpkg with a single model.

         (Will be deprecated: if there is one remain parameter, that is model file)

Examples:
    model2nnpkg.sh -m add.tflite                                              => create nnpackage 'add' in ./
    model2nnpkg.sh -o out -m add.tflite                                       => create nnpackage 'add' in out/
    model2nnpkg.sh -o out -p addpkg -m add.tflite                             => create nnpackage 'addpkg' in out/
    model2nnpkg.sh -c add.cfg -m add.tflite                                   => create nnpackage 'add' with add.cfg
    model2nnpkg.sh -o out -p addpkg -m a1.tflite a2.tflite -i a1.json a2.json => create nnpackage 'addpkg' with models a1.tflite and a2.tflite in out/

```
