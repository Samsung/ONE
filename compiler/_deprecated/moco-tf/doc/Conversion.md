This document outlines how to express each TensorFlow operation on top of _loco_

**CAUTION** All the python examples below are written in Python 3 with TensorFlow v1.13.

**DISCLAIMER** _loco_ does not support named values, but all the below _loco_ examples assign "name" to each value to make it easy to read.

### Placeholder

**Placeholder** in _TensorFlow_ corresponds to **Pull** in _loco_.

_Python_:
```python
import tensorflow as tf
input = tf.placeholder(dtype=tf.float32, shape=[3, 4], name='input')
print(tf.get_default_graph().as_graph_def())
```

API reference: [tf.placeholder](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf)

_TensorFlow_
```prototext
node {
  name: "input"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 3 }
        dim { size: 4 }
      }
    }
  }
}
```

_loco_:
```
%input = Pull(dtype: FLOAT32, shape: [3, 4])
Push(%input)
```

### Identity

**Identity** in _TensorFlow_ corresponds to **Forward** in _loco_.

_Python_:
```python
import tensorflow as tf
input = tf.placeholder(dtype=tf.float32, shape=[3, 4])
ident = tf.identity(input)
print(tf.get_default_graph().as_graph_def())
```

API reference: [tf.identity](https://www.tensorflow.org/api_docs/python/tf/identity)

_TensorFlow_:
```
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim { size: 3 }
        dim { size: 4 }
      }
    }
  }
}
node {
  name: "Identity"
  op: "Identity"
  input: "Placeholder"
  attr {
    key: "T"
    value { type: DT_FLOAT }
  }
}
```

_loco_:
```
%input = Pull(dtype: FLOAT32, shape: [3, 4])
%ident = Forward(%input)
Push(%ident)
```

### Const

**Const** in _TensorFlow_ corresponds to **ConstGen** in _loco_.

_Python_:
```python
import tensorflow as tf
constant = tf.constant(value=[1.0], dtype=tf.float32, shape=[3, 4])
tf.get_default_graph().as_graph_def()
```

API reference: [tf.constant](https://www.tensorflow.org/versions/r1.13/api_docs/python/tf/constant)

_TensorFlow_:
```
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value { type: DT_FLOAT }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim { size: 3 }
          dim { size: 4 }
        }
        float_val: 1.0
      }
    }
  }
}
```

_loco_:
```
%constant = ConstGen(dtype: FLOAT32, shape: [3, 4], data: ...);
Push(%constant)
```
