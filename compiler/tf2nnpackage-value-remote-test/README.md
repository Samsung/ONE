# tf2nnpackage-value-remote-test

`tf2nnpackage-value-remote-test` does random value test for nnpackage file using remote machine, normally Odroid, which `onert` runs on.

### Prerequisites

1. Tensorflow library
    - Make sure that Tensorflow library could be found at `nncc configure` step. If there is no Tensorflow library, this test will not be created.
    - If CMake reports TensorFlow library is not found in configure step, even when the library exists, set [`TENSORFLOW_PREFIX`](../../infra/cmake/packages/TensorFlowConfig.cmake#1) to include Tensorflow library like below.
        ```sh
        $ ./nncc configure -DTENSORFLOW_PREFIX=/path/to/Tensorflow/library
        ```
    - `TENSORFLOW_PREFIX` should contain Tensorflow library as shown below.
        ```
        TENSORFLOW_PREFIX
            ├ include
            |  ├ tensorflow
            |  |     └ c
            |  |       ├ c_api.h
            |  ├ ...
            |
            ├ lib
            |  ├ libtensorflow.so
            |  ├ ...
            ├ ...
        ```
1. Runtime Library and Binary files
    - Detailed information is located in [here](../../docs/howto/how-to-cross-build-runtime-for-arm.md)
    - If you build runtime, related files will be produced in `Product/out`. Do not rename or move it.
1. Remote machine information and test list
    - You should create `test.lst` file first as shown below.
        - Set IP address and username of remote machine using `set` command.
        - Add recipes which you want to verify. They are in `ONE/res/TensorFlowLiteRecipes/`
        ```cmake
        #--------------- Remote Machine Setting ---------------#
        set(REMOTE_IP "xxx.xxx.xxx.xxx")
        set(REMOTE_USER "remote_username")
        
        #--------------------- Tests list ---------------------#
        add(Abs_000)
        add(ArgMax_000)
        ...
        ```
    - If any recipe is added, or if `REMOTE_IP` and `REMOTE_USER` is not given, `tf2nnpackage-value-remote-test` will not be created. 
1. (Optional) ssh authentication
    - This test uses `ssh` and `scp` commands, and those commands require a password of remote machine whenever they are called. This means that you should enter the password everytime when `ssh` and `scp` require.
    - This test resolves the problem by using `ssh-copy-id`, which copies the public key of host machine to `authorized_keys` of remote machine. Because of that, this test will ask the password of remote machine only once, at the first time. This is the only user interaction while running this test.
    - If you do not want to interact with system, just do `ssh-copy-id ${REMOTE_USER}@${REMOTE_IP}` in advance, before running this test. Once `ssh-copy-id` is done, there will be no user-interaction action while running the test.

### Running

- If you finished prerequisites properly, configuring -> building -> testing steps create cmake test automatically.
- All the related materials will be sent to `REMOTE_WORKDIR` in remote machine. Default value of `REMOTE_WORKDIR` is `CVT_YYMMDD_hhmmss`, which means Circle Value Test done on YY/MM/DD at hh:mm:ss.
- `REMOTE_WORKDIR` will not be removed automatically after this test finish.
    ```sh
    $ ./nncc configure && ./nncc build

    # Default REMOTE_WORKDIR is CVT_YYMMDD_hhmmss folder
    $ ./nncc test -R tf2nnpackage_value_remote_test

    # You can set REMOTE_WORKDIR where you have write privilege
    $ REMOTE_WORKDIR=/path/you/want/ ./nncc test -R tf2circle_value_pbtxt_remote_test
    ```

### Generated Files While Running

- All related files(`pb`, `circle`, `h5` ... etc.) are taken from `build/compiler/common-artifacts` folder.
- `nnpkg_test.sh`, runtime products and each nnpackage are sent to `REMOTE_WORKDIR` in remote machine.
- Each test result is generated in `build/compiler/common-artifacts` with the name `${RECIPE}.log`

### Check Test Result

- Summary of test result will be created as csv file in host.
    ```sh
    # Result_latest is symbolic link to the latest csv result file
    # Print the latest test result
    $ cat build/compiler/tf2circle-value-pbtxt-remote-test/Result_latest
    TEST_NAME, TF2CIRCLE, CIRCLE_VALUE_TEST
    UNIT_Add_000, TRUE, TRUE
    ...

    # List all result csv files
    $ ls build/compiler/tf2circle-value-pbtxt-remote-test/Result_*.csv
    Result_20191119_212521.csv
    ...
    ```
