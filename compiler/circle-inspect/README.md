# circle-inspect

_circle-inspect_ allows users to retrieve various information from a Circle model file

## Information to inspect

Operators with `--operators`
- show operator codes one line at a time in execution order

Example
```
$ circle-inspect --operators model.circle
```

Result
```
RESHAPE
DEPTHWISE_CONV_2D
ADD
```

To get the count of specific operator, use other tools like sort, uniq, etc.
