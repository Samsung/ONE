# large graph splitter

Large pb file is hard to test.
This tool splitters large pb file intp small pb files.

## `test5.py`
- test code to load pb and duplicate an existing op with newly created placeholder(s) as its input(s)

## `test6.py`
- prints name, shape, value of a specific op

## `test7.py`
- generates test graphs (a.k.a., tgraph)
- tested with graph having control flow ops
- placeholder shape should be read from the result of `test6.py`

## `test8.py`
- save input and output tensors of tgraph into hdf5 file

## other files
- only for reference
