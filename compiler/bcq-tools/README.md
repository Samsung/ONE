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
