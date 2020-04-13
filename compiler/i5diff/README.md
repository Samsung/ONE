# i5diff

_i5diff_ compares two HDF5 files that _nnkit_ HDF5 export action generates.

**DISCLAIMER** _i5diff_ is not designed as a general diff tool.
It works only for HDF5 files that _nnkit_ HDF5 export action generates.

## Yet Another Diff?

_i5diff_ is able to detect _shape mismatch_ that _h5diff_ cannot detect.

To be precise, _h5diff_ is also able to detect _shape mismatch_.
Unfortunately, however, _h5diff_ ends with 0 exitcode in the presence of _shape mismatch_, and thus
it is impossible to use _h5diff_ for continuous integration.

## How to use

```
$ /path/to/i5diff -d 0.001 /path/to/fst.h5 /path/to/snd.h5
```
