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

Session session = new Session("/sdcard/nnpkg/model", "cpu;acl_neon;acl_cl");
session.prepare();

Tensor[] inputs = session.prepareInputs();
session.setInputs(inputs);

Tensor[] outputs = session.prepareOutputs();
session.setOutputs(inputs);

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
