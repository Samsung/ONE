## Example

``` java
import com.samsung.onert.Session;

Session session = new Session("/sdcard/nnpkg/model", "cpu;acl_neon;acl_cl");
session.prepare();

Bytebuffer[] inputs = null; // ... fill out inputs
Bytebuffer[] outputs = null;

session.run(inputs, outputs);

session.close();
```
