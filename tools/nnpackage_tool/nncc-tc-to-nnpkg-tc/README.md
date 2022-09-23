# nncc-tc-to-nnpkg-tc

`nncc-tc-to-nnpkg-tc` is a tool to convert nncc testcase to nnpackage testcase.

It takes `nncc-tc` as input and generates `nnpkg-tc`.

## Usage

```
Usage: nncc-tc-to-nnpkg-tc.sh [options] nncc_tc_name
Convert nncc testcase to nnpackage testcase.

Options:
    -h   show this help
    -i   set input directory (default=.)
    -o   set nnpackage testcase output directory (default=.)

Env:
   model2nnpkg    path to model2nnpkg tool (default={this_script_home}/../model2nnpkg)

Examples:
    nncc-tc-to-nnpkg-tc.sh -i build/compiler/tf2tflite UNIT_Add_000
      => create nnpackage testcase in ./ from build/compiler/tf2tflite/UNIT_Add_000.*
    nncc-tc-to-nnpkg-tc.sh -o out UNIT_Add_000
      => create nnpackage testcase in out/ using ./UNIT_Add_000.*
```
