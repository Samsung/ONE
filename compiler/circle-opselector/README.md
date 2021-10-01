# circle-opselector

`circle-opselector` is a tool for creating new circle model by various methods to select nodes.

## Example

### 1. Select from location numbers

```bash
./circle-opselector --by_id "1-3,5" --select input.circle output.circle
```

Then, output.circle which has node 1, 2, 3 and 5 will be created.

### 2. Select from node names

```bash
./circle-opselector --by_name --select "Add_1,Sub_1,Concat_2" input.circle output.circle
```


Then, output.circle which has node Add_1, Sub_1 and Concat_2 will be created.

### 3. Select nodes without several nodes

```bash
./circle-opselector --by_id "1-3,5" --deselect input.circle output.circle
```

Then, output.circle which except node 1, 2, 3 and 5 will be created.
