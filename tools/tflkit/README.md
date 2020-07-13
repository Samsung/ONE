# tflkit

## Purpose

There are a lot of tools related to TfLite. However, it is inconvenient to use the tools directly because there are many locations and parameters. The tflkit has been created to make it easier to run frequently used tools in scripts. The function provided in this directory uses existing tools rather than implementing them directly. So, additional modifications may occur depending on the TensorFlow version or other external factors. The function provided in this directory will be gradually expanded.

## Prerequisites

The scripts here use TensorFlow's tools, so you need an environment to build TensorFlow.
Running the scripts within this tutorial requires:
* [Install Bazel](https://docs.bazel.build/versions/master/install.html), the build tool used to compile TensorFlow.

Initially, no external packages are installed on this project. Therefore, before running these scripts, you should install the associcated packages by running the following command once.
```
make configure
```

## Summarize TF model

### TensorFlow

TensorFlow uses `summarize_graph` tool to inspect the model and provide guesses about likely input and output nodes, as well as other information that's useful for debugging. For more information, see [Inspecting Graphs](https://github.com/tensorflow/tensorflow/tree/9590c4c32dd4346ea5c35673336f5912c6072bf2/tensorflow/tools/graph_transforms#inspecting-graphs) page.

Usage:
```
$ bazel build tensorflow/tools/graph_transforms:summarize_graph
$ bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=<pb file>
```

### tflkit

Usage:
```
$ ./summarize_pb.sh <pb file>
```

The results shown below:
```
$ ./summarize_pb.sh inception_v3.pb

	inception_v3.pb

Inputs
	name=input
	type=float(1)
	shape=[?,299,299,3]
Outputs
	name=InceptionV3/Predictions/Reshape_1, op=Reshape
Op Types
	488 Const
	379 Identity
	95 Conv2D
	94 FusedBatchNorm
	94 Relu
	15 ConcatV2
	10 AvgPool
	4 MaxPool
	2 Reshape
	1 BiasAdd
	1 Placeholder
	1 Shape
	1 Softmax
	1 Squeeze

	14 Total
```

## Summarize TfLite model

### tflkit

Usage:
```
$ ./summarize_tflite.sh <tflite file>
```

The results shown below:
```
$ ./summarize_tflite.sh inception_v3.tflite 
[Main model]

Main model input tensors: [317]
Main model output tensors: [316]

Operator 0: CONV_2D (instrs: 39,073,760, cycls: 39,073,760)
	Fused Activation: RELU
	Input Tensors[317, 0, 5]
		Tensor  317 : buffer  183 |  Empty | FLOAT32 | Shape [1, 299, 299, 3] (b'input')
		Tensor    0 : buffer  205 | Filled | FLOAT32 | Shape [32, 3, 3, 3] (b'InceptionV3/Conv2d_1a_3x3/weights')
		Tensor    5 : buffer   52 | Filled | FLOAT32 | Shape [32] (b'InceptionV3/InceptionV3/Conv2d_1a_3x3/Conv2D_bias')
	Output Tensors[6]
		Tensor    6 : buffer  285 |  Empty | FLOAT32 | Shape [1, 149, 149, 32] (b'InceptionV3/InceptionV3/Conv2d_1a_3x3/Relu')

[...]

Operator 125: SOFTMAX (instrs: 4,003, cycls: 4,003)
	Input Tensors[225]
		Tensor  225 : buffer  142 |  Empty | FLOAT32 | Shape [1, 1001] (b'InceptionV3/Logits/SpatialSqueeze')
	Output Tensors[316]
		Tensor  316 : buffer   53 |  Empty | FLOAT32 | Shape [1, 1001] (b'InceptionV3/Predictions/Reshape_1')


Number of all operator types: 6
	CONV_2D                               :   95 	 (instrs: 11,435,404,777)
	MAX_POOL_2D                           :    4 	 (instrs: 12,755,516)
	AVERAGE_POOL_2D                       :   10 	 (instrs: 36,305,334)
	CONCATENATION                         :   15 	 (instrs: 0)
	RESHAPE                               :    1 	 (instrs: ???)
	SOFTMAX                               :    1 	 (instrs: 4,003)
Number of all operators                       :  126 	 (total instrs: 11,484,469,630)
```

## Convert a TensorFlow model into TfLite model

### TensorFlow

TensorFlow provides some kinds of converting guideline. In Python, the [TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter) class will help you to convert a TensorFlow GraphDef or SavedModel into `output_format` using TOCO. The `output_format` can be `TFLITE` or `GRAPHVIZ_DOT` format. The default `output_format` is `TFLITE`. And there is a Python command line interface for running TOCO, and its name is [tflite_convert](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/tflite_convert.py). This converts a TensorFlow GraphDef or SavedModel into `TFLITE` or `GRAPHVIZ_DOT` format like [TFLiteConverter](https://www.tensorflow.org/api_docs/python/tf/lite/TFLiteConverter). These two way also supports to convert a TensorFlow Keras model into `output_format`. Both functions are implemented using a tool called TOCO.

### with tflkit

The tflkit uses the [tflite_convert](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/python/tflite_convert.py) python command line interface to convert a TensorFlow model into TfLite model. It only supports to convert a TensorFlow GraphDef file into `TFLITE` format file. This tool supports the creation of individual `TFLITE` files for different input shapes. When converting to multiple `TFLITE` files, it needs to put a string called `NAME` in `TFLITE_PATH`. The string `NAME` will be replaced by what is listed in the `NAME` environment. This tool requires an information file as a parameter. There is an [example file](convert.template) for a convert information. The `--tensorflow_path` and `--tensorflow_version` can change the TensorFlow location. By default, it uses `externals/tensorflow` directory.

Convert information:
  * GRAPHDEF_PATH : Full filepath of file containing frozen TensorFlow GraphDef.
  * TFLITE_PATH : Full filepath of the output TfLite model. If `NAME` optional environment is used, it must include `NAME` string in the file name. (ex. `[...]/inception_v3_NAME.tflite`)
  * INPUT : Names of the input arrays, comma-separated.
  * INPUT_SHAPE : Shapes correspoding to `INPUT`, colon-separated.
                  For the creation of individual `TFLITE` files for different input shapes, space-separated.
  * OUTPUT : Names of the output arrays, comma-seperated.
  * NAME(Optional) : Names of the individual `TFLITE` files for different input shapes, space-seperated.

Usage (for example, [InceptionV3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)):
```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
$ tar xzvf inception_v3_2018_04_27.tgz ./inception_v3.pb
$ cat > convert.info << END
GRAPHDEF_PATH="${PWD}/inception_v3.pb"
TFLITE_PATH="${PWD}/inception_v3.tflite"
INPUT="input"
INPUT_SHAPE="1,299,299,3"
OUTPUT="InceptionV3/Predictions/Reshape_1"
#NAME=""
END
$ ./tflite_convert.sh --info=./convert.info --tensorflow_version=1.12
$ ls *.tflite
inception_v3.tflite
```

Usage (for example, multiple `TFLITE` files):
```
$ cat > convert_multiple.info << END
GRAPHDEF_PATH="${PWD}/inception_v3.pb"
TFLITE_PATH="${PWD}/inception_v3_NAME.tflite"
INPUT="input"
INPUT_SHAPE="1,299,299,3 3,299,299,3"
OUTPUT="InceptionV3/Predictions/Reshape_1"
NAME="batch1 batch3"
END
$ ./tflite_convert.sh --info=./convert_multiple.info --tensorflow_version=1.12
$ ls *.tflite
inception_v3_batch1.tflite
inception_v3_batch3.tflite
```

## Optimize a TensorFlow model for inference

### TensorFlow

This [optimize tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py) behaves like a transform tool. However, this tool is optimized to convert the trained TensorFlow graph for inference. This tool removes parts of a graph that are only needed for training. These include:
  - Removing training-only operations like checkpoint saving.
  - Stripping out parts of the graph that are never reached.
  - Removing debug operations like CheckNumerics.
  - Folding batch normalization ops into the pre-calculated weights.
  - Fusing common operations into unified version.
The input and output file of this tool is a TensorFlow GraphDef file.

### with tflkit

The [optimize_for_inference.sh](optimize_for_inference.sh) file invokes the TensorFlow [optimize tool](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py). This tool requires a optimize information file as a parameter. Here is an [example file](optimize.template) for this tool. The information file needs `INPUT` and `OUTPUT` array names. The [summarize_pb.sh](summarize_pb.sh) file will help you to define the `INPUT` and `OUTPUT` array names. The `--tensorflow_path` can change the TensorFlow location. By default, it uses `externals/tensorflow` directory.

Optimize information:
  * GRAPHDEF_PATH : Full filepath of file containing frozen TensorFlow GraphDef.
  * OPTIMIZE_PATH : Full filepath to save the output optimized graph.
  * INPUT : Names of the input arrays, comma-separated.
  * OUTPUT : Names of the output arrays, comma-seperated.

Usage (for example, [InceptionV3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)):
```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
$ tar xzvf inception_v3_2018_04_27.tgz ./inception_v3.pb
$ cat > optimize.info << END
GRAPHDEF_PATH="${PWD}/inception_v3.pb"
OPTIMIZE_PATH="${PWD}/inception_v3.optimize.pb"
INPUT="input"
OUTPUT="InceptionV3/Predictions/Reshape_1"
END
$ ./optimize_for_inference.sh --info=./optimize.info
$ ls *.pb
inception_v3.optimize.pb  inception_v3.pb
```

## Transform a TensorFlow graph

### TensorFlow

The trained TensorFlow model can be trasformed by some variants to deploy it in production. This [Graph Transform Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#graph-transform-tool) provides to support this behavior. There are so many transform options in this tool. For more information on transform options, please see [this page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#transform-reference). The input and output file of this tool is a TensorFlow GraphDef file.

### with tflkit

The [transform_graph.sh](transform_graph.sh) file supports to transform a TensorFlow GraphDef using various transform options. This tool requires a transform information file as a parameter and the transform options are described in the information file. There is an [example file](transform.template) for this tool. The information file needs `INPUT` and `OUTPUT` array names. The [summarize_pb.sh](summarize_pb.sh) file will help you to define the `INPUT` and `OUTPUT` array names. The `--tensorflow_path` can change the TensorFlow location. By default, it uses `externals/tensorflow` directory.

Transform information:
  * GRAPHDEF_PATH : Full filepath of file containing frozen TensorFlow GraphDef.
  * TRANSFORM_PATH : Full filepath of the output TensorFlow GraphDef.
  * INPUT : Names of the input arrays, comma-separated.
  * OUTPUT : Names of the output arrays, comma-seperated.
  * TRANSFORM_OPTIONS : Names of transform option, space-separated.
                        By default, it includes the following options.
    * strip_unused_nodes : Removes all nodes not used in calculated the layer given in `OUTPUT` array, fed by `INPUT` array.
    * remove_nodes : Removes the given name nodes from the graph.
      * `Identity` is not necessary in inference graph. But if it needs in inference graph, this tool does not remove this node.
      * `CheckNumerics` is useful during training but it is not necessary in inference graph.
    * fold_constants : Replaces the sub-graps that always evaluate to constant expressions with those constants. This optimization is always executed at run-time after the graph is loaded, so it does'nt help latency, but it can simplify the graph and so make futher processing easier.
    * fold_batch_norms : Scans the graph for any channel-wise multiplies immediately after convolutions, and multiplies the convolution's weights with the Mul instead so this can be omitted at inference time. It should be run after `fold_constants`.

Usage (for example, [InceptionV3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)):
```
$ wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
$ tar xzvf inception_v3_2018_04_27.tgz ./inception_v3.pb
$ cat > transform.info << END
GRAPHDEF_PATH="${PWD}/inception_v3.pb"
TRANSFORM_PATH="${PWD}/inception_v3.transform.pb"
INPUT="input"
OUTPUT="InceptionV3/Predictions/Reshape_1"
TRANSFORM_OPTIONS="strip_unused_nodes(type=float, shape=\"1,299,299,3\") remove_nodes(op=Identity, op=CheckNumerics) fold_constants(ignore_errors=true) fold_batch_norms"
END
$ ./transform_graph.sh --info=./transform.info
$ ls *.pb
inception_v3.pb  inception_v3.transform.pb
```

## Freeze a TensorFlow model

### TensorFlow

TensorFlow provides methods to save and restore models. Each method stores related files in different ways. Here are two common ways to save the freeze stored models.
  1. Use [tf.train.Saver](https://www.tensorflow.org/guide/saved_model#save_and_restore_variables)
    In this way, it creates a `MetaGraphDef` file and checkpoint files that contain the saved variables. Saving this way will result in the following files in the exported directory:

      ```
      $ ls /tmp/saver/
      checkpoint  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
      ```

  2. Use [SavedModel](https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel)
    It is the easiest way to create a saved model. Saving this way will result in the following files in the exported directory:

      ```
      $ ls /tmp/saved_model/
      saved_model.pb  variables
      $ tree /tmp/saved_model/
      /tmp/saved_model/
      ├── saved_model.pb
      └── variables
          ├── variables.data-00000-of-00001
          └── variables.index
      ```

The [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) tool receives these files as input parameters and combines the stored variables and the standalone graphdef to generate a frozen graphdef file.

### with tflkit

The tflkit provides the simple way to create a frozen graph using [freeze_graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) tool. This tool requires an information file as a parameter. There is an [example file](freeze.template) for a freeze tool. Either `SAVED_MODEL` or `META_GRAPH` must be declared. And `META_GRAPH` is always used with `CKPT_PATH`. The `--tensorflow_path` can change the TensorFlow location. By default, it uses `externals/tensorflow` directory.

Freeze information:
  * SAVED_MODEL : Full directory path with TensorFlow `SavedModel` file and variables.
  * META_GRAPH : Full filepath of file containing TensorFlow `MetaGraphDef`.
  * CKPT_PATH : Full filepath of file containing TensorFlow variables. (ex. [...]/*.ckpt)
  * FROZEN_PATH : Full filepath to save the output frozen graph.
  * OUTPUT : Names of the output arrays, comma-separated.

Usage (for example, `tf.train.Saver`):
```
$ cat > sample_saver.py << END
import tensorflow as tf

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/saver/model.ckpt")
  print("Model saved in path: %s" % save_path)
END
$ python sample_saver.py
$ ls /tmp/saver/
checkpoint  model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
$ cat > freeze_saver.info << END
#SAVED_MODEL=""                                  
META_GRAPH="/tmp/saver/model.ckpt.meta"
CKPT_PATH="/tmp/saver/model.ckpt"
FROZEN_PATH="/tmp/saver/model.frozen.pb"
OUTPUT="v2"
END
$ ./freeze_graph.sh --info=./freeze_saver.info
$ ls /tmp/saver/*.pb
/tmp/saver/model.frozen.pb
```

Usage (for example, `SavedModel`):
```
$ cat > sample_saved_model.py << END
import tensorflow as tf

# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  tf.saved_model.simple_save(sess, "/tmp/saved_model", inputs={'v1':v1}, outputs={'v2':v2})
END
$ python sample_saved_model.py
$ ls /tmp/saved_model/
saved_model.pb  variables
$ cat > freeze_saved_model.info << END
SAVED_MODEL="/tmp/saved_model"
#META_GRAPH=
#CKPT_PATH=
FROZEN_PATH="/tmp/saved_model/model.frozen.pb"
OUTPUT="v2"
END
$ ./freeze_graph.sh --info=./info/freeze_saved_model.info
$ ls /tmp/saved_model/
model.frozen.pb  saved_model.pb  variables
$ ls /tmp/saved_model/*.frozen.pb
/tmp/saved_model/model.frozen.pb
```
