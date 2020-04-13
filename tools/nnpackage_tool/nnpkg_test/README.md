# nnpkg_test

`nnpkg_test` is a tool to run an nnpackage testcase.

`nnpackage testcase` is an nnpackage with additional data:

- input.h5 (input data)
- expected.h5 (expected outpute data)

`nnpkg_test` uses `nnpackage_run` internally to run `nnpackage`.

Then, it compares through `difftool` (either `i5diff` or `h5diff`).

`nnpkg_test` returns `0` on success, `non-zero` otherwise.

## Usage

```
$ tools/nnpackage_tool/nnpkg_test/nnpkg_test.sh -h
Usage: nnpkg_test.sh [options] nnpackage_test
Run an nnpackage testcase

Returns
     0       success
  non-zero   failure

Options:
    -h   show this help
    -i   set input directory (default=.)
    -o   set output directory (default=.)
    -d   delete dumped file on failure.
         (dumped file are always deleted on success) (default=0)

Environment variables:
   nnpackage_run    path to nnpackage_run (default=Product/out/bin/nnpackage_run)
   difftool         path to i5diff or h5diff (default=h5diff)

Examples:
    nnpkg_test.sh Add_000                => run ./Add_000 and check output
    nnpkg_test.sh -i nnpkg-tcs Add_000   => run nnpkg-tcs/Add_000 and check output

```
