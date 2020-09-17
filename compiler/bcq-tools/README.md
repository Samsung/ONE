# BCQ Tools

This directory includes some tools related with BCQ.

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

First of all, metadata will be generated as following description.
```
< Generated Metadata in BCQ version 1 >
[0] Starting magic number                = {-2e9 + 27}
[1] Version of BCQ                       = {1}
[2] The number of original model outputs = {N | N > 0}
[3] Bundle size                          = {7, 8}
[4] Ending magic number                  = {2e9 - 27}
```
- BCQ version 1
    - Two magic numbers, starting and ending magic number, are used for indicating that the node includes BCQ metadata. To decrease value duplication probability, prime number is used and the value is inserted not only at the beginning but also at the end.
    - The word **bundle** means that a set of BCQ information and BCQ applicable operation. If six BCQ information nodes are used for one operation, the six information nodes and the other one operation are packaged as **bundle**. Then, in this case, the bundle size will be 6 + 1 = 7.

### Caution

- If there is no BCQ information in original model, any changes will be applied.
