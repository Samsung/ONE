# onecc-docker

`onecc-docker` supports `one-cmd`, which is currently only supported by ubuntu 18.04, in more OS.



## Description

For now, `one-cmds` tools only support Ubuntu 18.04 and 20.04(not officially).
So people in other environments can't use our tools unless they upgrade the OS (or install Ubuntu OS).

Therefore, we developed `onecc` that runs using a docker so that users can use `one-cmds` more easily. This is `onecc-docker.`

> Currently, `onecc-docker` is only supported in Linux environments.



## Requirements

- Linux OS

- Docker

    - `onecc-docker` requires the current `user ID` to be included in the `docker group` because it requires the docker-related commands to be executed without `sudo` privileges.

        ```
        sudo usermod -aG docker ${USER}
        ```

- Python 3.8



## Structure

### Summary 

`onecc-docker` creates a docker image with the latest release version of `ONE`, runs `onecc` inside the docker container after the container is driven to deliver the desired result to the user.

### To check the latest version of ONE

- We use github api to get the latest release version of ONE. Create a docker file using the imported version.

### Check Docker Image Presence and Build Image

- To prevent repetitive image generation, we verify that there is an `onecc` image of the latest `one` release version.
- If not, build the docker image using the docker file.

### Run Docker Container

- When the above processes are successfully completed, we drive the docker container with the command the user wants. `onecc` runs on the user's current path and returns the result to the command line.



## Examples

### Preparations

We prepare the Tensorflow file, configure file, and workflow for the test.

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

