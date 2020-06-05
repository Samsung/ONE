import tensorflow as tf

'''
  input       axis
    |          |
  +--+--+      |
  |   shape    |
  |    |       |
  reshape      |
    |          |
    |          |
     expand_dim     1
            |       |
           expand_dim
                |
              relu
                |
              tanh
                |
              relu6
                |

'''

input_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 2, 3), name="Hole")
axis_ = tf.compat.v1.placeholder(dtype=tf.int32, shape=(1), name="HoleAxis")
inshape_ = tf.compat.v1.shape(input_, name="Shape")
reshape_ = tf.compat.v1.reshape(input_, inshape_, name="Reshape")
exdims01_ = tf.compat.v1.expand_dims(reshape_, axis_, name="ExpandDims1")
exdims02_ = tf.compat.v1.expand_dims(exdims01_, 1, name="ExpandDims2")
relu_ = tf.compat.v1.nn.relu(exdims02_, name="Relu")
tanh_ = tf.compat.v1.nn.tanh(relu_, name="Tanh")
relu6_ = tf.compat.v1.nn.relu(tanh_, name="Relu6")

'''
eric@labs:~/work/A/test  ( ) (12:01:50)
$ export MODEL=PBSplitterTest1
eric@labs:~/work/A/test  ( ) (12:01:50)
$ docker run -v /home/eric/:/home/eric -w /home/eric/work/A/ONE/res/TensorFlowPythonModels tensorflow/tensorflow:1.15.0-py3 python3 tfpem.py ${MODEL}
Generate 'PBSplitterTest1.pbtxt'
Generate 'PBSplitterTest1.pbtxt' - Done
eric@labs:~/work/A/test  ( ) (12:02:46)
$ mv -f /home/eric/work/A/ONE/res/TensorFlowPythonModels/PBSplitterTest1.pbtxt .
eric@labs:~/work/A/test  ( ) (12:03:40)
$ cat PBSplitterTest1.pbtxt | ~/work/1/ONE/build/compiler/tfkit/tfkit encode > PBSplitterTest1.pb
'''