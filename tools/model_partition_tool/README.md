# Heuristic Graph Partitioning

This folder contains the necessary scripts to perform a a heuristic-based graph partitioning for machine learning models.

The main contents of this folder are as follows:

- [Python Files](#python-scripts)
- [How to Run Partitioning Algorithm?](#how-to-partition-tflite-model)
- [Example Script](#example-script-to-generate-partition-map)


## Python Scripts
The python scripts (**python3**) require an installation of Tensorflow 2.x package to retrieve TFLite model operations. Additionally, please ensure that the python `numpy` package has been installed beforehand. The scripts also import the following modules: `queue`, `json` and `argparse`, all of which should be available by default. If not, please install them either by `pip install <package>` or `sudo apt install python-<package>`.

`Graph.py` is the main script that processes the model graph topology and implements the partitioning algorithm. Correspondingly, there are two classes within, namely `GraphTopology` and `GraphPartition`. `GraphTopology` has a container `GraphPartition` object within.

`graph_analysis.py` is a helper module for translating TFLite models to graph data structures. `graph_analysis.py` is imported inside
`Graph.py`.


## How To Partition TFLite Model?
To partition a TFLite model, simply `import Graph` at the outset. There are two ways to run the partitioning algorithm. If you prefer quick results without having to inspect the intermediate results, follow the steps below:

### Quick Run For Final Result
To get the partitioning result quickly, follow the steps below:

1. Create a `GraphTopology` object as shown below:
```
In [70]: g = Graph.GraphTopology('inceptionV3.tflite', 'inceptionV3.chrome.json')
```
**Note**: Here, the argument `inceptionV3.chrome.json` is a single-execution trace of operation execution times, and is obtained using the Chrome Trace profiler.

2. Run the **MinMax** partitioning algorithm over the topology. Specify the number of partitions (K) and the number of topological orderings (nruns) to evaluate before settling for the best result.
```
In [71]: g.partition_minmax_multiple(K=4, nruns=10)

INFO:Topology:Session ids: 
INFO:Topology:Partition 0 : [0, 1, 2, 3, 4, 5, 6, 13, 7, 10, 14, 11, 12, 8, 9, 15, 22, 16, 23, 19, 17, 20, 21], sum weight = 292393
INFO:Topology:Partition 1 : [18, 24, 26, 27, 28, 25, 29, 30, 31, 32, 33, 34, 38, 35, 36, 37, 39, 49, 44, 41, 45, 42, 50, 46, 43, 40, 47, 48, 51, 53, 56, 52], sum weight = 293959
INFO:Topology:Partition 2 : [61, 57, 58, 54, 55, 62, 59, 60, 63, 73, 74, 65, 64, 68, 66, 69, 67, 70, 71, 72, 75, 76, 80, 77, 78, 79, 81, 82], sum weight = 290835
INFO:Topology:Partition 3 : [83, 84, 85, 86, 87, 90, 94, 91, 88, 89, 92, 93, 95, 96, 101, 97, 106, 98, 102, 103, 104, 99, 105, 107, 100, 108, 109, 110, 111, 114, 119, 120, 112, 115, 116, 117, 113, 118, 121, 122, 123, 124, 125], sum weight = 293819
INFO:Topology:Session Graph:
[[  0   1   0   0]
 [  0   0   1   0]
 [  0   0   0   1]
 [  0   0   0   0]]
INFO:Topology:Edge cut: 12
INFO:Topology:Memory overhead (bytes): 4366144

In [72]: 
```

### Detailed View
For a detailed breakdown of the runtime steps, execute the function calls shown below:

1. Create a `GraphTopology` object:
```
In [70]: g = Graph.GraphTopology('inceptionV3.tflite', 'inceptionV3.chrome.json')
```

2. Perform a topological sort
```
In [73]: g.topological_sort()
```


3. Partition the graph into K sub-graphs, using the topological order obtained above
```
In [74]: g.partition_graph(K=4)
```

4. View the execution time of each partition
```
In [75]: g.partition_summary()
INFO:Minmax:Session Graph:
[[  0   1   0   0]
 [  0   0   1   0]
 [  0   0   0   1]
 [  0   0   0   0]]
INFO:Minmax:Partition 0 : [0, 1, 2, 3, 4, 5, 6, 13, 8, 7, 14, 9, 10, 11, 12, 15, 22, 23, 17, 16, 18, 19], sum weight = 276635
INFO:Minmax:Partition 1 : [20, 21, 24, 26, 28, 27, 29, 25, 30, 31, 32, 33, 38, 35, 36, 34, 37, 39, 40, 41, 44, 45, 46, 42, 43, 49, 50, 47, 48, 51, 52, 61], sum weight = 299334
INFO:Minmax:Partition 2 : [56, 53, 54, 57, 58, 55, 59, 60, 62, 63, 73, 65, 66, 67, 68, 69, 74, 70, 64, 71, 72, 75, 80, 81, 77, 76, 78, 82, 85], sum weight = 291593
INFO:Minmax:Partition 3 : [83, 86, 84, 79, 87, 94, 90, 91, 88, 92, 89, 93, 95, 106, 107, 96, 97, 101, 102, 98, 104, 103, 99, 100, 105, 108, 114, 109, 119, 120, 110, 112, 111, 113, 115, 117, 116, 118, 121, 122, 123, 124, 125], sum weight = 303444
 ```

 5. Run a *OneShot* version of the partitioning algorithm
```
In [90]: indx, minmax, edge_cnt, memory_overhead = g.partition_minmax(oneshot=True)
INFO:Minmax:Bottleneck id --> 3, sorted neighbors --> [(2, 291593)]
DEBUG:Topology:====Moving from session 3 to session 2
INFO:Minmax:[old maxval] 303444 --> 300754 [new maxval], id: 86
WARNING:Minmax:Candidate 87 cannot be moved, as it violates acyclic constraint
WARNING:Minmax:Move rejected, max value is greater
DEBUG:Topology:====Successful move from session 3 to session 2
INFO:Minmax:Bottleneck id --> 2, sorted neighbors --> [(3, 294283), (1, 299334)]
DEBUG:Topology:====Moving from session 2 to session 3
WARNING:Minmax:Move rejected, max value is greater
DEBUG:Topology:====Failed move from session 2 to session 3
DEBUG:Topology:====Moving from session 2 to session 1
WARNING:Minmax:Move rejected, max value is greater
DEBUG:Topology:====Failed move from session 2 to session 1
INFO:Minmax:Session Graph:
[[  0   1   0   0]
 [  0   0   1   0]
 [  0   0   0   1]
 [  0   0   0   0]]
INFO:Minmax:Partition 0 : [0, 1, 2, 3, 4, 5, 6, 13, 8, 7, 14, 9, 10, 11, 12, 15, 22, 23, 17, 16, 18, 19], sum weight = 276635
INFO:Minmax:Partition 1 : [20, 21, 24, 26, 28, 27, 29, 25, 30, 31, 32, 33, 38, 35, 36, 34, 37, 39, 40, 41, 44, 45, 46, 42, 43, 49, 50, 47, 48, 51, 52, 61], sum weight = 299334
INFO:Minmax:Partition 2 : [56, 53, 54, 57, 58, 55, 59, 60, 62, 63, 73, 65, 66, 67, 68, 69, 74, 70, 64, 71, 72, 75, 80, 81, 77, 76, 78, 82, 85, 86], sum weight = 300754
INFO:Minmax:Partition 3 : [83, 84, 79, 87, 94, 90, 91, 88, 92, 89, 93, 95, 106, 107, 96, 97, 101, 102, 98, 104, 103, 99, 100, 105, 108, 114, 109, 119, 120, 110, 112, 111, 113, 115, 117, 116, 118, 121, 122, 123, 124, 125], sum weight = 294283
```

**Note** Please set debug levels in the script accordingly, for example, `g.set_dbglevel(logging.DEBUG)`.

## Example Script To Generate Partition Map
An example script `test_parition.py` is added to the folder. Please run `python3 test_partition.py --help` for details. The script parses the TFLite model file and the trace JSON as arguments, and creates a `partition_map.json` at the same location as the TFLite file. An output from running `test_partition.py` is shown below:

```
$ python3 test_partition.py /tmp/nnpackage/inception_v3/inception_v3.tflite /tmp/inceptionV3.chrome.json --num_parts=4

...
...
INFO:Topology:Partition 0 : [0, 1, 2, 3, 4, 5, 6, 8, 13, 7, 9, 14, 10, 11, 12, 15, 19, 17, 16, 20, 21, 22, 23], sum weight = 292393
INFO:Topology:Partition 1 : [18, 24, 28, 31, 32, 29, 30, 25, 26, 27, 33, 35, 34, 38, 36, 37, 39, 49, 40, 44, 45, 41, 50, 42, 43, 46, 47, 48, 51, 52, 56, 57], sum weight = 296611
INFO:Topology:Partition 2 : [53, 61, 54, 58, 59, 60, 62, 55, 63, 68, 65, 73, 66, 64, 69, 67, 74, 70, 71, 72, 75, 80, 76, 81, 85, 82, 83, 77, 86], sum weight = 286608
INFO:Topology:Partition 3 : [78, 79, 84, 87, 94, 90, 91, 88, 89, 92, 93, 95, 96, 106, 101, 102, 104, 107, 103, 97, 99, 105, 98, 100, 108, 114, 119, 120, 115, 117, 110, 116, 118, 112, 109, 111, 113, 121, 122, 123, 124, 125], sum weight = 295394
INFO:Topology:Session Graph:
[[  0   1   0   0]
 [  0   0   1   0]
 [  0   0   0   1]
 [  0   0   0   0]]
INFO:Topology:Edge cut: 12
INFO:Topology:Memory overhead (bytes): 4403136

$ cat /tmp/nnpackage/inception_v3/partition_map.json
{"partition_map": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], "num_partitions": 4}
$
```