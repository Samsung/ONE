# BCQ Tools

This directory includes some tools related with BCQ.

## preserve_bcq_info

### Purpose

`preserve_bcq_info` is for preserving constant nodes which include BCQ information.
When `.pb` file is converted to `.tflite` file by TFlite converter, constant nodes whose values are exactly same are removed and then linked to only one representative node.
This makes us impossible to know what constant node should be linked to a node which we want to apply BCQ.
One of the solutions is making all the same constant nodes different by inserting unique values and ignore the newly generated unique values when BCQ fusing is applied.
`preserve_bcq_info` will generate and insert unique dummy values to the constant nodes whose values are same not to be removed by Tensorflow Lite converter.
As a result, BCQ information will be preserved.

### How to use

```bash
preserve_bcq_info \
--input_path /path/to/original_model.pb \
--output_path /path/to/preserved_model.pb
```

### How it works

If we add unique dummy value at the end of each constant nodes, all the constant nodes would be different. Following is an example.

```
[Original Constant Nodes]
const(value=[1, 2, 3], name='const1')
const(value=[1, 2, 3], name='const2')
const(value=[1, 2, 3], name='const3')

[After BCQ information preserved]
const(value=[1, 2, 3, -1], name='const1')
const(value=[1, 2, 3, -2], name='const2')
const(value=[1, 2, 3, -3], name='const3')
```

For dummy values, negative values are used instead of positive values.
This is because positive valus may be confused with original constant node values.
For your information, unique dummy value starts from -1 and moves to -2, -3, ..., -N, where N is the number of preserved constant nodes.

### Caution

- Newly generated dummy values should be ignored when the constant nodes are used.

## generate_bcq_output_arrays

### Purpose

To apply BCQ, BCQ information nodes should be designated as model output so that they are alive even after TFLite conversion is finished.
However, there are so many nodes to designate and sometimes we cannot copy and paste all of them because the string size is too big.
`generate_bcq_output_arrays` is for generating output_arrays, which include BCQ information nodes.

### How to use

```bash
generate_bcq_output_arrays \
--input_path /path/to/original_model.pb \
--output_path /path/to/output_arrays.txt
```

### How it works

```
[Original BCQ information nodes]
const(value=[1, 2, 3, -1], name='const1')
const(value=[1, 2, 3, -2], name='const2')
const(value=[1, 2, 3, -3], name='const3')

[Generated output_arrays]
,const1,const2,const3
```

### Caution

- Generated output_arrays will be start with comma.

## generate_bcq_metadata

### Purpose

To apply BCQ, additional information which is not included in original model is needed.
For example, without metadata, we cannot know from where to where BCQ related nodes are included. Metadata for BCQ is needed due to those reason and `generate_bcq_metadata` will generate metadata in the model according to each BCQ version.

### How to use

```bash
generate_bcq_metadata \
--input_path /path/to/original_model.pb \
--output_path /path/to/metadata_inserted_model.pb
--output_arrays output1,output2,...,outputN
```

### How it works

Metadata will be generated as following description.
```
< Generated Metadata in BCQ version 1 >
[0] Starting magic number                = {-2e9 + 27}
[1] Version of BCQ                       = {1}
[2] The number of original model outputs = {N | N > 0}
[3] Bundle size                          = {7, 8}
[4] Ending magic number                  = {2e9 - 27}
```
- BCQ version 1
    - Two magic numbers, starting and ending magic number, are used for indicating that the model includes BCQ metadata. To decrease value duplication probability, prime number is used and the value is inserted not only at the beginning but also at the end.
    - The word **bundle** means that a set of BCQ information and BCQ applicable operation. If six BCQ information nodes are used for one operation, the six information nodes and the other one operation are packaged as **bundle**. Then, in this case, the bundle size will be 6 + 1 = 7.

### Caution

- If there is no BCQ information in original model, any changes will be applied.
