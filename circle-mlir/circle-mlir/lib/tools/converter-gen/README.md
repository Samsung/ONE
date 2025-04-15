# About converter-gen

converter_gen is from TensorFlow
- `tensorflow/compiler/mlir/lite/converter_gen.cc`
- commit hash 2a3f646e6177178fde3e79f15b582580252e558c

build script is translated from bazel `BUILD`
```bazel
tf_native_cc_binary(
    name = "converter-gen",
    srcs = [
        "converter_gen.cc",
    ],
    compatible_with = get_compatible_with_cloud(),
    deps = [
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TableGen",
        "@llvm-project//mlir:TableGen",
    ],
)
```
