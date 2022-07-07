## How to create

```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
$ tar -zxf mobilenet_v1_1.0_224.tgz

$ python tools/tflitefile_tool/select_operator.py mobilenet_v1_1.0_224.tflite <( echo 0-1 ) mv1.0_1.tflite
$ python tools/tflitefile_tool/select_operator.py mv1.0_1.tflite <( echo 0 ) mv1.0.tflite
$ python tools/tflitefile_tool/select_operator.py mv1.0_1.tflite <( echo 1 ) mv1.1.tflite

# make sure three tflite is valid
$ ./Product/out/bin/tflite_comparator mv1.0_1.tflite
$ ./Product/out/bin/tflite_comparator mv1.0.tflite
$ ./Product/out/bin/tflite_comparator mv1.1.tflite

$ tools/nnpackage_tool/model2nnpkg/model2nnpkg.sh -m mv1.0.tflite mv1.1.tflite -p two_tflites
$ cat two_tflites/metadata/MANIFEST
{
  "major-version" : "1",
  "minor-version" : "2",
  "patch-version" : "0",
  "configs"     : [  ],
  "models"      : [ "mv1.0.tflite", "mv1.1.tflite" ],
  "model-types" : [ "tflite", "tflite" ]
}

# update minor-version, and add additional fields manually
```
