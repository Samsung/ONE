# C-Sharp TFLite API Directory structure
```
.
├── packaging
│   ├── TFLiteSharp.manifest
│   └── TFLiteSharp.spec
├── README.md
├── TFLiteNative
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── tflite_log.h
│   │   └── tflite_nativewrapper.h
│   ├── src
│   │   └── tflite_nativewrapper.cpp
│   └── tflite-native.pc.in
├── TFLiteSharp
│   ├── TFLiteSharp
│   │   ├── src
│   │   │   └── Interpreter.cs
│   │   └── TFLiteSharp.csproj
│   └── TFLiteSharp.sln
└── TFLiteSharpTest
    ├── TFLiteSharpTest
    │   ├── Program.cs
    │   └── TFLiteSharpTest.csproj
    └── TFLiteSharpTest.sln
```

# Build C-Sharp TFLite
gbs should be used to build TFLiteSharp package. nnfw is also built by gbs. As in most cases when building nnfw we won't intend to build TFLiteSharp hence we have separated its build process, so in order to build TFLiteSharp below command is needed:
```
nnfw$ gbs build --packaging-dir=contrib/TFLiteSharp/packaging/ --spec=TFLiteSharp.spec -A armv7l
```
This will first build the TFLiteNative package containing native c++ bindings between c# api and tflite api
and then it will build TFLiteSharp(c# api package).

Please use gbs.conf file corresponding to tizen image version. In most cases gbs.conf file should be same as the one which is used to build nnfw.
# C-Sharp TFLite API list

## Interpreter Class

### Constructor

The `Interpreter.cs` class drives model inference with TensorFlow Lite.

#### Initializing an `Interpreter` With a Model File

The `Interpreter` can be initialized with a model file using the constructor:

```c#
public Interpreter(string modelFile);
```

Number of threads available to the interpereter can be set by using the following function:
```c#
public void SetNumThreads(int num_threads)
```

### Running a model

If a model takes only one input and returns only one output, the following will trigger an inference run:

```c#
interpreter.Run(input, output);
```

For models with multiple inputs, or multiple outputs, use:

```c#
interpreter.RunForMultipleInputsOutputs(inputs, map_of_indices_to_outputs);
```

The C# api also provides functions for getting the model's input and output indices given the name of tensors as input:

```c#
public int GetInputIndex(String tensorName)
public int GetOutputIndex(String tensorName)
```

Developer can also enable or disable the use of NN API based on hardware capabilites:
```c#
public void SetUseNNAPI(boolean useNNAPI)
```

### Releasing Resources After Use

An `Interpreter` owns resources. To avoid memory leak, the resources must be
released after use by:

```c#
interpreter.Dispose();
```
