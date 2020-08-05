# Custom Operators

This document explains about custom operator and how custom op is represented in nnpackage.

## What is custom operator?

Custom operator(hereafter custom op) is used to provide a new operator implementation.
It can be anything that does not exist in current runtime implementation.

You can use custom operator for several use cases, possible use cases are:

- when an operator in tensorflow is not supported in nnfw runtime
- when an operator is supported, however, you would like to use your own implementation
  - it may be for optimization, by grouping several operators into one super operator.

## Custom op in model

nnpackage will support several kinds of models.
Currently the only type is tflite.

### tflite

If you're using `tflite` format, it is same format to tensorflow lite.

You can generate `tflite` model with custom op using `tflite_convert`.
Please find the documentation in tensorflow official site.

## Custom op kernel implementation

You need to provide the kernel of custom op in the following form:

```
/*
 * Custom kernel evaluation function
 *
 * param[in] params custom operation parameters
 * param[in] userdata pointer to user-specified buffer( kernel instance specific )
 */
typedef void (*nnfw_custom_eval)(nnfw_custom_kernel_params *params, char *userdata,
                                 size_t userdata_size);

```

The structures and relevant APIs are defined in nnfw APIs.
Please see `nnfw_experimental.h` for detail.

You can find example in `nnfw` repository.

Custom op kernel implementation is stored in nnpackage in form of prebuilt library.

It is example nnpackage structure for `FillFrom`:

```
FillFrom
├── FillFrom.tflite
├── custom_op
│   ├── libFillFrom.armv7l-linux.debug.a
│   └── libFillFrom.armv7l-linux.release.a
└── metadata
    └── MANIFEST
```

All custom operator libraries are put under `{nnpackage_root}/custom_op/lib{customop_name}.{arch}-{os}.{buildtype}.a`.

## How to use custom op in app

To use custom op, the app has to register the operators with `nnfw_register_custom_op_info`.


```
/*
 * custom operation registration info
 */
typedef struct
{
  nnfw_custom_eval eval_function;
} custom_kernel_registration_info;

NNFW_STATUS nnfw_register_custom_op_info(nnfw_session *session, const char *id,
                                         custom_kernel_registration_info *info)
```

Please find sample app in `nnfw` repository

The `id` should be unique in an app.

