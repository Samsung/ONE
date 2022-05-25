### Introduce `tflite-infer` tool

It's used to infer tflite model and get the result data.
`tflite-infer` code itself is included in [#9122](https://github.com/Samsung/ONE/pull/9122) too. This pr includes tests, manpage, cmake codes.


### `tflite-infer` I/O file strategy

Working on `tflite-infer`, options for loading and dumping I/O data are the most difficult part to design. Here are the restrictions and requirements that I thinking about.

1. Each I/O data needs to support every `npy`, `h5` and `bin` format. 
2. After dumping randomly generated data as a file, it should be able to be loaded without any additional process. 
3. Dumping data should not mess up the working directory with many independent files.
4. (Optional) Data file should be expandable for multiple data inference, for example, infer multiple images in one `tflite-infer` command.

Here are two command line examples which show the changes between before and after `tflite-infer` called. 

<details><summary>Case of loading npy input data</summary>

```console
$ tree .
.
├── model.tflite
└── simple_data
    ├── simple_data.input.0.npy
    └── simple_data.input.1.npy

$ tflite-infer --loadable model.tflite \
    --input-spec  npy:simple_data \
    --dump-input-npy  simple_new \
    --dump-input-h5   simple_new \
    --dump-output-npy simple_new \
    --dump-output-h5  simple_new

$ tree .
.
├── model.tflite
│── simple_data
│   ├── simple_data.input.0.npy
│   └── simple_data.input.1.npy
├── simple_new
│   ├── simple_new.input.0.npy
│   ├── simple_new.input.1.npy
│   └── simple_new.output.0.npy
├── simple_new.input.h5
└── simple_new.output.h5
```

</details>
</br>
<details><summary>Case of loading h5 input data</summary>

```console
$ tree .
.
├── model.tflite
└── simple_data.input.h5

$ tflite-infer --loadable model.tflite \
    --input-spec  h5:simple_data \
    --dump-input-npy  simple_new \
    --dump-input-h5   simple_new \
    --dump-output-npy simple_new \
    --dump-output-h5  simple_new

$ tree .
.
├── model.tflite
└── simple_data.input.h5
├── simple_new
│   ├── simple_new.input.0.npy
│   ├── simple_new.input.1.npy
│   └── simple_new.output.0.npy
├── simple_new.input.h5
└── simple_new.output.h5
```

</details>
</br>

### test design description

Category | Test # | Description
:--: | :--: | :--
positive test | P-01 | -h printing test
positive test |	P-02	|-l --input-spec npy:\<filename> execution & output npy dump
positive test |	P-03	|-l --input-spec h5:\<filename> execution & output h5 dump
positive test |	P-04	|-l --input-spec any input npy&h5 shape/dtype test
positive test|	P-05	|-l --input-spec non-zero input npy&h5 value, shape/dtype test
positive test|	P-06	|-l --input-spec positive input npy&h5 value, shape/dtype test
negative test |	N-01 |	-l option is missing
negative test |	N-02 | 	--input-spec is missing
negative test |	N-03 |	--input-spec \<option> is invalid
negative test |	N-04 |	-l model is not found
negative test |	N-05 |	--input-spec npy:\<filename> file is not found
