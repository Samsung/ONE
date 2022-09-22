# Tutorial

This document provides a tutorial for running _onecc-docker_.

### Preparations

Prepare source files like below;

```
$ tree
.
├── Dockerfile
├── inception_v3.pb
├── inception_v3.tflite
├── onecc-docker
├── onecc.template.cfg
└── onecc.workflow.json

0 directories, 6 files
```

#### `-h` ,`--help`

##### run

```
$ ./onecc-docker -h
usage: onecc [-h] [-v] [-C CONFIG] [-W WORKFLOW] [-O OPTIMIZATION] [COMMAND <args>]

Run ONE driver via several commands or configuration file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -V, --verbose         output additional information to stdout or stderr
  -C CONFIG, --config CONFIG
                        run with configuation file
  -O OPTIMIZATION       optimization name to use (No available optimization
                        options)
  -W WORKFLOW, --workflow WORKFLOW
                        run with workflow file

compile to circle model:
  import                Convert given model to circle
  optimize              Optimize circle model
  quantize              Quantize circle model

package circle model:
  pack                  Package circle and metadata into nnpackage

run backend tools:
  codegen               Code generation tool
  profile               Profile backend model file
  infer                 Infer backend model file
```

#### `-v`, `--version`

##### run

```
$ ./onecc-docker -v
onecc version 1.21.0
Copyright (c) 2020-2022 Samsung Electronics Co., Ltd. All Rights Reserved
Licensed under the Apache License, Version 2.0
https://github.com/Samsung/ONE
```

#### `-C CONFIG`, `--config CONFIG`

##### onecc.template.cfg

```
[onecc]
one-import-tf=False
one-import-tflite=True
one-import-bcq=False
one-import-onnx=False
one-optimize=True
one-quantize=False
one-parition=False
one-pack=True
one-codegen=False

[one-import-tflite]
# mandatory
; tflite file
input_path=inception_v3.tflite
; circle file
output_path=inception_v3.circle

[one-optimize]
input_path=inception_v3.circle
output_path=inception_v3.opt.circle
generate_profile_data=False

[one-pack]
input_path=inception_v3.opt.circle
output_path=inception_v3_pack
```

##### run

```
$ ./onecc-docker -C onecc.template.cfg 
model2nnpkg.sh: Generating nnpackage inception_v3.opt in inception_v3_pack
```

##### tree

```
$ tree
.
├── Dockerfile
├── inception_v3.circle
├── inception_v3.circle.log
├── inception_v3.opt.circle
├── inception_v3.opt.circle.log
├── inception_v3.pb
├── inception_v3.tflite
├── inception_v3_pack
│   └── inception_v3.opt
│       ├── inception_v3.opt.circle
│       └── metadata
│           └── MANIFEST
├── inception_v3_pack.log
├── onecc-docker
├── onecc.template.cfg
└── onecc.workflow.json
```



#### `-W WORKFLOW`, `--workflow WORKFLOW`

##### onecc.workflow.json

```
$ cat onecc.workflow.json 
{
    "workflows": [
        "MY_WORKFLOW"
    ],
    "MY_WORKFLOW": {
        "steps": [
            "IMPORT_TF",
            "OPTIMIZE",
            "PACK"
        ],
        "IMPORT_TF": {
            "one-cmd": "one-import-tf",
            "commands": {
                "input_path": "inception_v3.pb",
                "output_path": "inception_v3.circle",
                "input_arrays": "input",
                "input_shapes": "1,299,299,3",
                "output_arrays": "InceptionV3/Predictions/Reshape_1",
                "converter_version": "v2"
            }
        },
        "OPTIMIZE": {
            "one-cmd": "one-optimize",
            "commands": {
                "input_path": "inception_v3.circle",
                "output_path": "inception_v3.opt.circle"
            }
        },
        "PACK": {
            "one-cmd": "one-pack",
            "commands": {
                "input_path": "inception_v3.opt.circle",
                "output_path": "inception_v3_pkg"
            }
        }
    }
}
```

##### run

```
$ ./onecc-docker -W onecc.workflow.json
Estimated count of arithmetic ops: 11.460 G  ops, equivalently 5.730 G  MACs
model2nnpkg.sh: Generating nnpackage inception_v3.opt in inception_v3_pkg
```

##### tree

```
$ tree
.
├── Dockerfile
├── inception_v3.circle
├── inception_v3.circle.log
├── inception_v3.opt.circle
├── inception_v3.opt.circle.log
├── inception_v3.pb
├── inception_v3.tflite
├── inception_v3_pkg
│   └── inception_v3.opt
│       ├── inception_v3.opt.circle
│       └── metadata
│           └── MANIFEST
├── inception_v3_pkg.log
├── onecc-docker
├── onecc.template.cfg
└── onecc.workflow.json

3 directories, 13 files
```

