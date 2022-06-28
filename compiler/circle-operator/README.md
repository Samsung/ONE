# circle-operator

_circle-operator_ allows users to retrieve operators information from a Circle model file

NOTE: this tool is primary for ONE-vscode where PartEditor needs names and codes
of the operators.

## Information with operators

Operators with `--name`
- show operator names one line at a time in execution order

Example
```
$ circle-operator --name model.circle
```

Result
```
conv1_pad/Pad
conv1_conv/BiasAdd
pool1_pad/Pad
```

Operators codes with `--code`
- show operator codes one line at a time in execution order

Example
```
$ circle-operator --name model.circle
```

Result
```
PAD
CONV_2D
PAD
```

Operators with both `--code` and `--name`
- show operator both codes and name separated with `,` one line at a time in execution order

Example
```
$ circle-operator --code --name model.circle
```

Result
```
PAD,conv1_pad/Pad
CONV_2D,conv1_conv/BiasAdd
PAD,pool1_pad/Pad
```

## Save to file

Use `--output_path` to save results to a file.

Example
```
$ circle-operator --name --output_path /tmp/result model.circle
```

Result
```
$ cat /tmp/result
conv1_pad/Pad
conv1_conv/BiasAdd
pool1_pad/Pad
```
