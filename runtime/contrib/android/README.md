## Example

``` java
import com.samsung.onert.Session;
import com.samsung.onert.Tensor;

Session session = new Session("/sdcard/nnpkg/model", "cpu;acl_neon;acl_cl");
session.prepare();

// init inputs and outputs
Tensor[] inputs = new Tensor[session.getInputSize()];
Tensor[] outputs = new Tensor[session.getOutputSize()];

// source inputs
// ...

session.setInputs(inputs);
session.setOutputs(outputs);

session.run();

// sink outputs

session.close();
```
