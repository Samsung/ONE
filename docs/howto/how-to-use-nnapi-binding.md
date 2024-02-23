# How to Use NNAPI Binding

Runtime supports [Android Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks) as a frontend. We provide the whole `NeuralNetworks.h` implementation. Its source code is located at `runtime/onert/api/nnapi`.

So users can just use NN API in the same way as [the official guide](https://developer.android.com/ndk/guides/neuralnetworks), but should watch the version of `NeuralNetworks.h`. It may not be the latest. There is a copy of the file is located at `runtime/nnapi-header/include/NeuralNetworks.h`.
