# About converter-gen

converter_gen is from TensorFlow
- `tensorflow/compiler/mlir/lite/converter_gen.cc`
- commit hash d5b57ca93e506df258271ea00fc29cf98383a374

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
