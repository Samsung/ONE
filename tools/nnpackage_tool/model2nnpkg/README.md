# model2nnpkg

`model2nnpkg` is a tool to convert model (either `tflite` or `circle`) to `nnpackage`.

It takes `modelfile` as input and generates `nnpackage`.

## Usage

```
Usage: model2nnpkg.sh [options] modelfile
Convert modelfile (either tflite or circle) to nnpackage.

Options:
    -h   show this help
    -o   set nnpackage output directory (default=.)
    -p   set nnpackage output name (default=[modelfile name])

Examples:
    model2nnpkg.sh add.tflite                  => create nnpackage 'add' in ./
    model2nnpkg.sh -o out add.tflite           => create nnpackage 'add' in out/
    model2nnpkg.sh -o out -p addpkg add.tflite => create nnpackage 'addpkg' in out/

```
