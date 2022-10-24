# TensorFlow Python Examples

## Prerequisite

- Python 3.8
- TensorFlow 2.8.0
- NOTE some examples may use old versions

## Directory Layout

```
tfpem.py <- TensorFlow Python Example Manager
examples/
  [EXAMPLE NAME]/
    __init__.py
```

## Folder naming convention

Follow python API name

## HOWTO: Create a Python environment

Install release debian packages in https://github.com/Samsung/ONE/releases
and enter virtual environment.
```
source /usr/share/one/bin/venv/bin/activate
```
You may have to prepare for the first time. Read [how-to-prepare-virtualenv.txt]
(https://github.com/Samsung/ONE/blob/master/compiler/one-cmds/how-to-prepare-virtualenv.txt)
for more information.

## HOWTO: Generate a pbtxt from examples

```
$ /path/to/python -B <path/to/tfpem.py> [EXAMPLE NAME 1] [EXANMPE NAME 2] ...
```

NOTE. Add "-B" option not to generate "__pycache__".

## HOWTO: Add a new example

TBA
