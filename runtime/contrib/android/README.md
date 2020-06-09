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
