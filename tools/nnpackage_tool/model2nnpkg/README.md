# model2nnpkg

`model2nnpkg` is a tool to convert model (e.g. `tflite`, `circle` or `tvn`) to `nnpackage`.

It takes `modelfile` as input and generates `nnpackage`.

## prerequisite

Python 3.5 or greater

## Usage

```
usage:  model2nnpkg.py [options]
  Examples:
      model2nnpkg.py -m add.tflite                           => create nnpkg "add" in current directory
      model2nnpkg.py -o out -m add.tflite                    => create nnpkg "add" in out/
      model2nnpkg.py -o out -p addpkg -m add.tflite          => create nnpkg "addpkg" in out/
      model2nnpkg.py -c add.cfg -m add.tflite                => create nnpkg "add" with add.cfg
      model2nnpkg.py -o out -p addpkg -m a1.tflite a2.tflite -i a1.json a2.json
        => create nnpkg "addpkg" with models a1.tflite and a2.tflite in out/


Convert model files (tflite, circle or tvn) to nnpkg.

options:
  -h, --help            show this help message and exit
  -o output_directory, --outdir output_directory
                        set nnpkg output directory
  -p nnpkg_name, --nnpkg-name nnpkg_name
                        set nnpkg output name (default=[1st modelfile name])
  -c conf [conf ...], --config conf [conf ...]
                        provide configuration files
  -m model [model ...], --models model [model ...]
                        provide model files
  -i io_info [io_info ...], --io-info io_info [io_info ...]
                        provide io info
```

## Usage (To be deprecated)
```
Usage: model2nnpkg.sh [options]
Convert modelfile (tflite, circle or tvn) to nnpackage.

Options:
    -h   show this help
    -o   set nnpackage output directory (default=.)
    -p   set nnpackage output name (default=[1st modelfile name])
    -c   provide configuration files
    -m   provide model files

Examples:
    model2nnpkg.sh -m add.tflite                                              => create nnpackage 'add' in ./
    model2nnpkg.sh -o out -m add.tflite                                       => create nnpackage 'add' in out/
    model2nnpkg.sh -o out -p addpkg -m add.tflite                             => create nnpackage 'addpkg' in out/
    model2nnpkg.sh -c add.cfg -m add.tflite                                   => create nnpackage 'add' with add.cfg
    model2nnpkg.py -o out -p addpkg -m a1.tflite a2.tflite => create nnpackage "addpkg" with models a1.tflite and a2.tflite in out/

```
