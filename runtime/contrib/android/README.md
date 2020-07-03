## How to setup env

Install android studio
- android studio: https://developer.android.com/studio

Install `ndk 20.0.5594570`
```
dragon@loki:~/Android/Sdk$ ./tools/bin/sdkmanager --install "ndk;20.0.5594570"
```

```
dragon@loki:~/Android/Sdk/ndk$ ls
20.0.5594570
```

Set `ANDROID_SDK_ROOT` or `ANDROID_HOME` for SDK
```
$ cat ~/.bashrc
...
export ANDROID_SDK_ROOT=$HOME/Android/Sdk
export ANDROID_HOME=$HOME/Android/Sdk
```

## How to build

after building `onert`
```
ONE/runtime/contrib/android $ ./gradlew build
...

ONE/runtime/contrib/android$ find . -name "*.aar"
./api/build/outputs/aar/com.samsung.onert-1.0-debug.aar
./api/build/outputs/aar/com.samsung.onert-1.0-release.aar
```

## Example

``` java
import com.samsung.onert.Session;
import com.samsung.onert.Tensor;

// for now, the only cpu backend has been supported
Session session = new Session("/sdcard/nnpkg/model/", "cpu");
session.prepare();

Tensor[] inputs, outputs;

// allocate inputs and outputs like below
//    int size = session.getInputSize();
//    inputs = new Tensor[size];
//    for (int i = 0; i < size; ++i){
//        TensorInfo ti = session.getInputTensorInfo(i);
//        inputs[i] = new Tensor(ti);
//    }
//    session.setInputs(inputs);

session.setInputs(inputs);
session.setOutputs(outputs);

// source inputs from outside
// inputs[i].buffer().put(outside_buffer);

session.run();

// sink outputs to inside
// outside_buffer.put(outputs[i].buffer());

session.close();
```

## How to add jni api
```
ONE/runtime/contrib/android $ ./update_jni_header.sh
```

and then follow code of `onert-native-api.h` on `onert-native-api.cpp`
