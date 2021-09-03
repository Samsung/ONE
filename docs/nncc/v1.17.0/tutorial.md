# Tutorial

Let's compile Inception_v3 model and make a circle model!

## Compile with onecc

This is the most simple way to compile and we recommend this way.

1. Download `nncc-1.17.0`.
    ```sh
    $ wget https://github.com/Samsung/ONE/releases/download/1.17.0/nncc-1.17.0.tar.gz
    $ tar -xvf nncc-1.17.0.tar.gz
    ```

1. Download pre-trained `inception_v3.pb` model file.
    ```sh
    $ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
    $ tar -xvf inception_v3_2018_04_27.tgz
    ```

1. Create configuration file as `inception_v3.cfg`.
    ```
    [onecc]
    one-import-tf=True
    one-import-tflite=False
    one-import-bcq=False
    one-optimize=True
    one-quantize=False
    one-pack=False
    one-codegen=False

    [one-import-tf]
    input_path=inception_v3.pb
    output_path=inception_v3.circle
    input_arrays=input
    input_shapes=1,299,299,3
    output_arrays=InceptionV3/Predictions/Reshape_1
    converter_version=v2

    [one-optimize]
    input_path=inception_v3.circle
    output_path=inception_v3.opt.circle
    O1=True
    ```

1. Prepare vitual environment.
    ```sh
    $ ./bin/one-prepare-venv
    ```

1. Compile using `onecc` and `inception_v3.cfg`
    ```sh
    $ ./bin/onecc -C inception_v3.cfg
    ```

1. Check that `inception_v3.opt.circle` is generated.
    ```sh
    $ ls -l inception_v3.opt.circle
    ---------- 1 user user 95342900  Month Date HH:MM inception_v3.opt.circle
    ```

## Compile step-by-step

Following procedures show how circle model is generated step-by-step.

1. Download `nncc-1.17.0`.
    ```sh
    $ wget https://github.com/Samsung/ONE/releases/download/1.17.0/nncc-1.17.0.tar.gz
    $ tar -xvf nncc-1.17.0.tar.gz
    ```

1. Download pre-trained `inception_v3.pb` model file.
    ```sh
    $ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
    $ tar -xvf inception_v3_2018_04_27.tgz
    ```

1. Prepare vitual environment.
    ```sh
    $ ./bin/one-prepare-venv
    ```

1. Create `inception_v3.tflite` using `inception_v3.pb` and `one-import-tf` with intermediate results.
    ```sh
    $ ./bin/one-import-tf \
    --input_path inception_v3.pb \
    --output_path inception_v3.circle \
    --input_arrays input \
    --input_shapes 1,299,299,3 \
    --output_arrays InceptionV3/Predictions/Reshape_1 \
    --converter_version v2 \
    --save_intermediate

    $ ls -l inception_v3.*
    ---------- 1 user user 95345492  Month Date HH:MM inception_v3.circle
    ---------- 1 user user 95345068  Month Date HH:MM inception_v3.tflite
    (...)
    ```
    - `one-import-tf` is executed using `bin/tf2tfliteV2.py` and `bin/tflite2circle`.
        - `tf2tfliteV2.py` : Generate `inception_v3.tflite` from `inception_v3.pb`
        - `tflite2circle` : Generate `inception_v3.circle` from `inception_v3.tflite`

1. Create `inception_v3.opt.circle` with `inception_v3.circle` and `one-optimize`.
    ```sh
    $ ./bin/one-optimize \
    --input_path inception_v3.circle \
    --output_path inception_v3.opt.circle \
    --O1

    $ ls -l inception_v3.opt.circle
    ---------- 1 user user 95342900  Month Date HH:MM inception_v3.opt.circle
    ```
    - `one-optimize` is executed using `bin/circle2circle`.
