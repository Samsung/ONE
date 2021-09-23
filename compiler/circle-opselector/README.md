# circle-opselector

`circle-opselector` is a tool for creating new circle model by various methods to select nodes.

## Example

### 1. Select from location numbers

```bash
./circle-opselector --by_id "1-3,5" --input input.circle --output output.circle
```

Then, output.circle which has node 1, 2, 3 and 5 will be created.

### 2. Select from node names

```bash
./circle-opselector --by_name "Add_1,Sub_1,Concat_2" --input input.circle --output output.circle
```

Then, output.circle which has node Add_1, Sub_1 and Concat_2 will created.
