# Tutorial

Let's compile Inception_v3 model and make a nnpackage!

## Prepare inception_v3 files

1. Download pre-trained `inception_v3.pb` model file.
    ```sh
    $ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
    $ tar -xvf inception_v3_2018_04_27.tgz
    ```
1. Create model information file as `inception_v3.info`.
    ```
    $ cat > inception_v3.info << "END"
    input,  input:0,  TF_FLOAT,  [1, 299, 299, 3]
    output, InceptionV3/Predictions/Reshape_1:0,  TF_FLOAT,  [1, 1001]
    END
    ```

## Let's compile inception_v3

1. Generate `nnpkg`. In this tutorial, let's generate to current directory.
    ```sh
    tf2nnpkg --use-tf2circle \
    --graphdef inception_v3.pb \
    --info inception_v3.info \
    -o .
    ```

## Check whether compilation is well done

- Check if all files are generated correctly.
    ```
    inception_v3
        ├ inception_v3.circle
        └ metadata
            └ MANIFEST
    ```
- Check if `MANIFEST` contents are correct.
    ```sh
    $ cat inception_v3/metadata/MANIFEST
    {
    "major-version" : "1",
    "minor-version" : "0",
    "patch-version" : "0",
    "models"      : [ "inception_v3.circle" ],
    "model-types" : [ "circle" ]
    }
    ```
