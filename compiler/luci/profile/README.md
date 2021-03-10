# luci-profile

`luci-profile` provides profiling related items.

## CircleNodeOrigin

`CircleNodeOrigin` allow us know where some node is originated from.

Let's assume following graph transformations are done.

```
    |                          |                         |
 [node1] --------+             |                         |
(id = 1)         |             |                         |
    |            +--------> [node5] ----------------> [node6]
    |            |     (origin = [1,2])          (origin = [1,2])
 [node2] --------+             |                         |
(id = 2)                       |                         |
    |                          |                         |
 [node3] -----------------> [node3] --------+-------> [node3]
(id = 3)                (origin = [3])      |    (origin = [3,4])
    |                          |            |            |
 [node4] -----------------> [node4] --------+            |
(id = 4)                (origin = [4])                   |
    |                          |                         |

<Circle1> -- optimizer --> <circle2> -- quantizer --> <circle3>
```

The most important purpose of using `CircleNodeOrigin` is preserving origin information.
Following changes show how origin information is preserved even after graph is transformed.

- `node3`
  - `node4` is absorbed to **existing** `node3`.
  - origin of `node4` is absorbed to origin of `node3`.
- `node5`
  - `node1` and `node2` are fused to **newly created** `node5`.
  - origin of `node1` and `node2` are inherited to origin of `node4`.
- `node6`
   - `node5` is **replaced with newly created** `node6`.
   - origin of `node5` is copied to origin of `node6`.

**Therefore, when using `CircleNodeOrigin`, please aware of the most important principle. "Preserve origin information"**

Next items are about implementation details to store the origin information.

### Source Table

Source table includes a set of id and name of origin node.

#### Binary format

```
[ entry_number : uint32_t ]
[ id : uint32_t ][ length : uint32_t ][ data : char * length ] * entry_number
```
- entry_number : The number of entries
  - Each entry consists of id, length, and data.
- id : ID of origin node
- length : Length of data
- data : Name of origin node **(null-terminated string)**

#### In-memory format
```cpp
// size = entry_number
std::map<uint32_t /* id */, std::string /* name */>
```
  - **`\0` chracter is not inserted to `std::string`.**

#### Example

Following example means "Name of origin 1 is node1".

```
[Binary Format]
 0x01 00 00 00 0x01 00 00 00 0x06 00 00 00 0x6e 0x6f 0x64 0x65 0x31 00
 ------------- ------------- ------------- ---------------------------
entry_number=1      id=1        length=6          data="node1\0"
```
```cpp
[In-memory Format]
std::map<uint32_t, std::string>({1, "node1"});
```

### Op Table

Op table includes a set of id of operation and id(s) of operation's origin nodes.

#### Binary format

Op table is stored in circle file as binary with following format.
```
[ entry_number : uint32_t ]
[ id : uint32_t ][ node_num : uint32_t ][ node_ids : uint32_t * node_num ] * entry_number
```
- entry_number : The number of entries
  - Each entry consists of id, node_num, and node_ids.
- id : ID of operation in circle model file
- node_num : The number of operation's origin nodes
- node_ids : Set of IDs of origin nodes

#### In-memory format
```cpp
std::map<uint32_t /* id */, std::set<uint32_t> /* node_ids */>
```

#### Example

Following example means "Operation 5 is originated from origin 1 and origin 2".

```
[Binary Format]
 0x01 00 00 00 0x05 00 00 00 0x02 00 00 00 0x01 00 00 00 0x02 00 00 00
 ------------- ------------- ------------- ---------------------------
entry_number=1      id=5       node_num=2        node_ids : 1, 2
```
```cpp
[In-memory Format]
std::map<uint32_t, std::set<uint32_t>>({5, std::set{1, 2}});
```
