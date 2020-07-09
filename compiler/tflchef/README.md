# tflchef

## What is tflchef?

Do you need a tensorflow lite model for testing? Ask it to _tflchef_.
Given a recipe, _tflchef_ will cook a tensorflow lite model for you.

**NOTE** A model that _tflchef_ generates is compatible with TensorFlow Lite in TensorFlow v1.13.1 release

## Tutorial: How to use?

This example explains how to generate a tensorflow lite model with a single Conv2D operation
with a kernel filled with random values generated according to normal (or gaussian) distribution (mean = 0.0f / stddev = 1.0f) and bias with constant values (1.1f) with _tflchef_.

The first step is to write a recipe!
Type the following command, and then you may get ``sample.recipe``:
```
$ cat > sample.recipe <<END
operand {
  name: "ifm"
  type: FLOAT32
  shape { dim: 1 dim: 3 dim: 3 dim: 2 }
}
operand {
  name: "ker"
  type: FLOAT32
  shape { dim: 1 dim: 1 dim: 1 dim: 2 }
  filler {
    tag: "gaussian"
    arg: "0.0"
    arg: "1.0"
  }
}
operand {
  name: "bias"
  type: FLOAT32
  shape { dim: 1 }
  filler {
    tag: "constant"
    arg: "1.1"
  }
}
operand {
  name: "ofm"
  type: FLOAT32
  shape { dim: 1 dim: 3 dim: 3 dim: 1 }
}
operation {
  type: "Conv2D"
  conv2d_options {
    padding: VALID
    stride_w: 1
    stride_h: 1
  }
  input: "ifm"
  input: "ker"
  input: "bias"
  output: "ofm"
}
input: "ifm"
input: "ker"
output: "ofm"
END
```

Generate ``sample.tflite`` from ``sample.recipe`` with one of the following commands:
- With redirection
```
$ cat sample.recipe | tflchef > sample.tflite
```
- Without redirection
```
$ tflchef-file sample.recipe sample.tflite
```

Done :)
