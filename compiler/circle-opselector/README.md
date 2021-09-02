# Circle Op-Selector

`Circle Op-Selector` is a tool for creating new circle model by various methods to select nodes .

## Example

### 1. Select from location numbers

```bash
./opselector --by_id "1-3,5" --input input.circle --output output.circle
```

Then, output.circle which has node 1, 2, 3 and 5 will created.

### 2. Select from node names

```bash
./opselector --by_name "Add_1,Sub_1,Concat_2" --input input.circle --output output.circle
```

Then, output.circle which has node Add_1, Sub_1 and Concat_2 will created.

## Options

⬜(Not Required) ✅(Required) ⬛️(Either is Required)
|Option|Argument|Description|Required|
|------|--------|-----------|--------|
|-h, --help|X|Print help message|⬜|
|--version|X|Print current program version|⬜|
|--input|[path to the file]|Insert target circle file src|✅|
|--output|[path to the file]|Insert ouput circle file src|✅|
|--by_id|[id1,id2] or [id1-id2|Insert operation id to select|⬛️|
|--by_name|[name1,name,...]|Insert operation name to select|⬛️|


## Directory Structure

```bash
compiler/circle-opselector/
    |
    +-- driver/
    |     +-- Driver.cpp
    |
    +-- CMakeLists.txt
    +-- README.md
    +-- requires.cmake
    |
    +-- pass/
          |
          +-- SinglePass.h
          |
          +-- function1.h
          +-- function1.cpp
          +-- function1.test.cpp
          |
					+-- ...
          
```
