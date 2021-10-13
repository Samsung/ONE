# circle-execution-plan

_circle-execution-plan_ tool provides model with "execution plan".

This tool takes circle file as input and returns modified circle file.
The output circle file contains plan (`CircleNodeMemoryPlan`) information for every node.


"execution plan" contains:
- number which determines order in which nodes will be executed
- memory offsets for node output tensors from the beginning of shared memory buffer

In order to record and read this metadata, we use `CircleImportMetadata` and `CircleExportMetadata`.
For this purpose we use `std::map<uint32_t, std::vector<uint32_t>> _memory_plan_table` which for each node with key ID contains encoded `CircleNodeMemoryPlan` data.

### Execution plan building

In order to build "execution plan" we use `ExecutionPlanner` class.
The main method is `get_execution_plan()` which for each node finds and writes to its annotations 
"execution plan". For this purpose there are two steps:
- determining the order of execution of nodes, which is stored in `_ordered_nodes` vector.
Now for this purpose there is only one default method `get_default_execution_order_plan()` that uses `loco::postorder_traversal(const std::vector<loco::Node *> &roots)`.
  In the future we can add new method and find the most suitable way to graph traversal.
  
- determining memory offsets for nodes from the beginning of shared memory buffer, which is stored in `_offsets`.
Now for this purpose there is one method `get_offsets_with_greedy_by_size()` that is the implementation of the "Greedy by Size" algorithm, which is described in https://arxiv.org/pdf/2001.03288.pdf article.
  The main objective is to minimize the size of the allocated memory block.
  In the future, other methods may also appear here to determine memory offsets for nodes
  in the best way.
