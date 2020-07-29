# BCQ Tools

This directory includes some tools related with BCQ.

## preserve_bcq_info

### Purpose

`preserve_bcq_info` is for preserving constant nodes which are including BCQ information.
When `.pb` file is converted to `.tflite` file by TFlite converter, constant nodes whose values are exactly same are removed and then linked to only one representative node.
It causes information missing problem because we don't know which constant nodes should be linked to even we still want to apply BCQ.
Therefore, we should preserve these BCQ information.

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

[After BCQ information preserving]
const(value=[1, 2, 3, -1], name='const1')
const(value=[1, 2, 3, -2], name='const2')
const(value=[1, 2, 3, -3], name='const3')
```

For dummy value, negative values are used instead of positive values.
This is because positive valus may be confused with original constant node values.
For your information, unique dummy value starts from -1 and moves to -2, -3, ..., -N.

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
