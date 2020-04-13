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

Examples:
    model2nnpkg.sh add.tflite        => create nnpackage in ./
    model2nnpkg.sh -o out add.tflite => create nnpackage in out/

```
