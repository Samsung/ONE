# Utils
Caffe model generation helpers

REQUIRES:

* caffe
* h5py
* lmdb
* numpy
* caffegen in `$PATH`

`GenerateCaffeModels.py` creates `*.prototxt` files for 1 and 2 layer caffe models
The generator can create multiple examples of any layer, assuming you add a
`how_many` field into the layer's dict. You will also need to replace the constants in said dict with `PH(type, param)` values, where `type` is the type of the placeholder variable
and `params` is a list (or tuple) of paramenters for generating the mock.

For an example of generating multiple instances of a layer see the `Log` layer. 

`Filler.sh`  fills a single model with random weights by using `caffegen` and creates a dir with a filled `prototxt` and a `caffemodel` binary file. The result directory is located in the same directory as the `prototxt` file

`AllFill.sh` fills all `*.prototxt` files in the current directory or in provided directory
(-d)
