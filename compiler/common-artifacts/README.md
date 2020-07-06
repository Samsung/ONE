# common-artifacts

`common-artifacts` is a module that produces intermediate artifacts that are commonly generated for compiler testing.

There are four modules used here.
- tflchef : recipe -> tflite
- tflite2circle : tflite -> circle (circlize)
- circle2circle : circle -> circle (optimize)
- TestDataGenerator : generate input.h5 and expected.h5 (tcgenerate)

## List of intermediate artifacts
- recipe
- tflite
- circle
- circle (applied all optimizations in circle2circle)
- input data for nnpackage (.h5)
- expected output data for nnpackage (.h5)

## How to exclude from resource generation
Sometimes a specific module that generates a resource does not support the generation of the resource, so exclusion is sometimes required.

There is a `exclude.lst` that performs the function. If you enter the name of steps(circlize, optimize, tcgenerate) and operator you want to exclude there, you can omit the module's step.

e.g.
```
$ cat exclude.lst
# circlize : Exclude from tflite-to-circle conversion(tflite2circle)

# optimize : Exclude from circle optimization(circle2circle)
optimize(ReLU6_000)
optimize(Where_000)
optimize(Where_001)

# tcgenerate : Exclude from test data generation(TestDataGenerator)
tcgenerate(Abs_000)
tcgenerate(AddN_000)
```
