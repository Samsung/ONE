/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "partition.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#define MAX_DEPTH 1000

/**
 * Prints the subgraph information of an ONNX model to specified files.
 *
 * @param subgraphs A vector containing subgraph information.
 * @param subgraphFileName The filename for the output of subgraph information.
 * @param otherSubgraphs A vector containing other subgraph information.
 * @param otherSubgraphFileName The filename for the output of other subgraph information.
 */
void PrintSubgraphs(std::vector<onnx::GraphProto> &subgraphs, char *subgraphFileName,
                    std::vector<onnx::GraphProto> &otherSubgraphs, char *otherSubgraphFileName)
{
  std::ofstream outFile(subgraphFileName);
  if (!outFile.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  int id = 0;
  for (const auto &vec : subgraphs)
  {
    outFile << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile << node.name() << " ";
    }

    id++;
    outFile << std::endl;
  }

  std::ofstream outFileOther(otherSubgraphFileName);
  if (!outFileOther.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  std::cout << "before:" << std::endl;
  for (const auto &vec : otherSubgraphs)
  {
    outFileOther << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFileOther << node.name() << " ";
    }

    id++;
    outFileOther << std::endl;
  }
}

/**
 * @brief     Constructs an adjacency list representation of the ONNX graph.
 *
 * @param     [in] g A const reference to an ONNX GraphProto object that contains the graph
 *            structure.
 * @param     [in,out] visited A pointer to an integer array used to mark whether nodes have been
 *             visited.
 * @pre       The 'visited' array should be pre-allocated with a size at least equal to the number
 *            of nodes in the graph.
 * @post      The 'visited' array will be initialized to 0 for all nodes.
 * @exception None
 * @return    A vector of GraphAdjacencyNode objects representing the adjacency list of the graph.
 */
std::vector<GraphAdjacencyNode> GetAdjancencyList(const onnx::GraphProto &g, int *visited)
{
  std::vector<GraphAdjacencyNode> adjacencyList;
  int nodeIndex = 0;
  for (const auto &node : g.node())
  {
    visited[nodeIndex] = 0;
    GraphAdjacencyNode adNode;
    adNode.index = nodeIndex;
    adNode.name = node.name();
    const auto &outputs = node.output();

    for (const auto &output : outputs)
    {
      int outputNodeIndex = 0;

      for (const auto &output_node : g.node())
      {
        int find_flag = 0;

        const auto &inputs = output_node.input();
        for (const auto &input : inputs)
        {
          if (output == input)
          {
            find_flag = 1;
            break;
          }
        }

        if (find_flag == 1)
        {
          if (std::find(adNode.outputNodeIndex.begin(), adNode.outputNodeIndex.end(),
                        outputNodeIndex) == adNode.outputNodeIndex.end())
          {
            adNode.outputNodeIndex.push_back(outputNodeIndex);
          }
        }

        outputNodeIndex++;
      }
    }

    nodeIndex++;
    adjacencyList.push_back(adNode);
  }

  return adjacencyList;
}

/**
 * @brief     Calculates the size of a specific node in the ONNX graph in kilobytes (KB).
 *
 * @param     [in] g A const reference to an ONNX GraphProto object that contains the graph
 * structure.
 * @param     [in] nodeIndex The index of the node for which the size is to be calculated.
 * @pre       The node_index should be a valid index within the range of nodes in the graph.
 * @post      None
 * @exception None
 * @return    The size of the node in kilobytes (KB).
 */
float CalculateNodeSize(const onnx::GraphProto &g, int nodeIndex) // unit : KB
{
  int64_t nodeSize = 0;
  for (int i = 0; i < g.node(nodeIndex).input_size(); i++)
  {
    std::string inputName = g.node(nodeIndex).input(i);

    for (int j = 0; j < g.initializer_size(); j++)
    {
      if (g.initializer(j).name() == inputName)
      {
        int64_t nodeInitSize = 4;

        for (int k = 0; k < g.initializer(j).dims().size(); k++)
        {
          nodeInitSize = g.initializer(j).dims(k) * nodeInitSize;
        }

        nodeSize += nodeInitSize;
        break;
      }
    }
  }
  return float(nodeSize * 1.0 / 1024.0);
}

/**
 * @brief     Depth-First Search (DFS) to build a NPU subgraph.
 *
 * @param     [in] g Input ONNX graph structure.
 * @param     [out] subgraph Output subgraph.
 * @param     [in,out] sugraphNodeIndex Vector storing indices of nodes in the subgraph.
 * @param     [in,out] visited Array recording whether nodes have been visited.
 * @param     [in] startNode Current starting node for the search.
 * @param     [in] nodeIndex Index of the current node.
 * @param     [in] adjacencyList Adjacency list representing connections between nodes in the graph.
 * @param     [in] supportOp List of supported operation types.
 * @param     [in] preferOp List of preferred operation types (not used in the code).
 * @param     [in] depthIn Current depth of the search.
 * @param     [in,out] graphSize Current size of the subgraph.
 * @param     [in] maxGraphSize Maximum allowed size of the subgraph.
 * @pre       `nodeIndex` should be a valid node index.
 * @post      If the subgraph size exceeds `maxGraphSize`, a warning message is printed.
 * @exception None
 */
void DFS(const onnx::GraphProto &g, onnx::GraphProto &subgraph, std::vector<int> &sugraphNodeIndex,
         int *visited, const onnx::NodeProto &startNode, int nodeIndex,
         std::vector<GraphAdjacencyNode> &adjacencyList, const std::vector<std::string> &supportOp,
         const std::vector<std::string> &preferOp, int depthIn, float &graphSize,
         float maxGraphSize)
{
  int depth_out = depthIn + 1;
  *subgraph.add_node() = startNode;
  visited[nodeIndex] = 1;
  sugraphNodeIndex.push_back(nodeIndex);
  float node_size = CalculateNodeSize(g, nodeIndex);
  graphSize += node_size;

  if (graphSize > maxGraphSize)
  {
    std::cout << "graph size exceed max size!" << graphSize << " " << maxGraphSize << std::endl;
  }

  for (int i = 0; i < int(adjacencyList[nodeIndex].outputNodeIndex.size()); i++)
  {
    if (i > 1)
    {
      std::cout << adjacencyList[nodeIndex].outputNodeIndex[i] << "->";
    }

    int next_nodeIndex = adjacencyList[nodeIndex].outputNodeIndex[i];
    const auto &next_node = g.node(next_nodeIndex);

    if (!visited[next_nodeIndex] &&
        (std::find(supportOp.begin(), supportOp.end(), next_node.op_type()) != supportOp.end()) &&
        (depth_out < MAX_DEPTH) && (graphSize < maxGraphSize))
    {
      DFS(g, subgraph, sugraphNodeIndex, visited, next_node, next_nodeIndex, adjacencyList,
          supportOp, preferOp, depth_out, graphSize, maxGraphSize);
    }
  }
}

/**
 * @brief     Perform a depth-first search (DFS) to build a CPU subgraph from a given starting node.
 *
 * @param     [in] g The original ONNX graph from which the subgraph will be extracted.
 * @param     [out] subgraph The subgraph being constructed.
 * @param     [out] sugraph_nodeIndex A vector to store indices of nodes included in the subgraph.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] startNode The starting node for the DFS.
 * @param     [in] nodeIndex The index of the starting node in the original graph.
 * @param     [in] adjacencyList The adjacency list representing the graph's structure.
 * @param     [in] depthIn The current depth of the DFS.
 * @param     [in,out] graphSize The cumulative size of the nodes in the subgraph.
 * @param     [in] maxGraphSize The maximum allowed size for the subgraph.
 *
 * @pre       The graph `g` and `adjacencyList` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       `graphSize` should be initialized to zero before the first call to this function.
 *
 * @post      The `subgraph` will contain the nodes visited during the DFS.
 * @post      The `sugraph_nodeIndex` will contain the indices of the nodes in the subgraph.
 * @post      The `visited` array will reflect the nodes that have been visited.
 * @post      The `graphSize` will reflect the cumulative size of the nodes in the subgraph.
 *
 * @exception None
 *
 * @return    None
 */
void DFSOther(const onnx::GraphProto &g, onnx::GraphProto &subgraph,
              std::vector<int> &sugraphNodeIndex, int *visited, const onnx::NodeProto &startNode,
              int nodeIndex, std::vector<GraphAdjacencyNode> &adjacencyList, int depthIn,
              float &graphSize, float maxGraphSize)
{
  int depth_out = depthIn + 1;
  *subgraph.add_node() = startNode;
  visited[nodeIndex] = 1;
  sugraphNodeIndex.push_back(nodeIndex);
  float node_size = CalculateNodeSize(g, nodeIndex);
  graphSize += node_size;

  if (graphSize > maxGraphSize)
  {
    std::cout << "graph size exceed max size!" << graphSize << " " << maxGraphSize << std::endl;
  }

  for (int i = 0; i < int(adjacencyList[nodeIndex].outputNodeIndex.size()); i++)
  {
    int next_nodeIndex = adjacencyList[nodeIndex].outputNodeIndex[i];
    const auto &next_node = g.node(next_nodeIndex);
    // do deep first search for each successor node
    if (!visited[next_nodeIndex] && (depth_out < MAX_DEPTH) && (graphSize < maxGraphSize))
      DFSOther(g, subgraph, sugraphNodeIndex, visited, next_node, next_nodeIndex, adjacencyList,
               depth_out, graphSize, maxGraphSize);
  }
}

/**
 * @brief     Determine and partition subgraphs from the given ONNX graph based on DFS strategy.
 * Compared with determine_subgraphs, this function is more stable but may produce more subgraphs
 *
 * @param     [out] subgraphs A vector to store the subgraphs that do not meet the preferred
 * operation criteria.
 * @param     [in] g The original ONNX graph to be partitioned.
 * @param     [out] otherSubgraphs A vector to store the subgraphs that do not meet the preferred
 * operation criteria.
 * @param     [in] d The device object containing information about supported and preferred
 * operations.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] adjacencyList The adjacency list representing the graph's structure.
 * @param     [in] strategy The partitioning strategy to be used (e.g., SPILTE_CPU_STRUCTURE_FIRST,
 * SPILTE_NPU_STRUCTURE_FIRST).
 *
 * @pre       The graph `g` and `adjacencyList` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       The device object `d` should be properly initialized with support and preferred
 * operations.
 *
 * @post      The `otherSubgraphs` vector will contain subgraphs that do not meet the preferred
 * operation criteria.
 * @post      The `visited` array will reflect the nodes that have been visited.
 *
 * @exception None
 *
 * @return    None
 */
void DetermineSubgraphsDFS(std::vector<onnx::GraphProto> &subgraphs, const onnx::GraphProto &g,
                           std::vector<onnx::GraphProto> &otherSubgraphs, Device &d, int *visited,
                           std::vector<GraphAdjacencyNode> &adjacencyList,
                           PartitionStrategy strategy)
{
  int maxSubgraphSize = d._max_subgraph_size;
  std::vector<std::string> supportOp;
  std::vector<std::string> preferOp;
  switch (strategy)
  {
    case SPILTE_CPU_STRUCTURE_FIRST:
    {
      supportOp = d.getCPUSupportOp();
      break;
    }
    case SPILTE_NPU_STRUCTURE_FIRST:
    {
      supportOp = d.getNPUSupportOp();
      preferOp = d.getNPUPreferOp();
      break;
    }
    default:
      break;
  }
  for (int i = 0; i < g.node_size(); i++)
  {
    if (!visited[i] &&
        (std::find(supportOp.begin(), supportOp.end(), g.node(i).op_type()) != supportOp.end()))
    {
      onnx::GraphProto subgraph;
      std::vector<int> sugraphNodeIndex;
      const auto &node = g.node(i);
      int depth = 0;
      float graphSize = 0;
      DFS(g, subgraph, sugraphNodeIndex, visited, node, i, adjacencyList, supportOp, preferOp,
          depth, graphSize, maxSubgraphSize);
      std::cout << "graphSize: " << graphSize << std::endl;
      int find_preferOp = 0;
      for (const auto &node : subgraph.node())
      {
        if (std::find(preferOp.begin(), preferOp.end(), node.op_type()) != preferOp.end())
        {
          find_preferOp = 1;
        }
      }
      if (find_preferOp)
      {
        subgraphs.push_back(subgraph);
      }
      else
      {
        for (const auto &index : sugraphNodeIndex)
        {
          visited[index] = 0;
        }
      }
    }
  }
  for (int i = 0; i < g.node_size(); i++)
  {
    if (!visited[i])
    {
      int depth = 0;
      float graphSize = 0;
      onnx::GraphProto subgraphOther;
      std::vector<int> sugraphNodeIndex;
      const auto &node = g.node(i);
      DFSOther(g, subgraphOther, sugraphNodeIndex, visited, node, i, adjacencyList, depth,
               graphSize, maxSubgraphSize);
      std::cout << "graphSize:" << graphSize << std::endl;
      otherSubgraphs.push_back(subgraphOther);
    }
  }
}

/**
 * @brief     Determine and partition subgraphs from the given ONNX graph using the index of nodes,
 * compared with DetermineSubgraphsDFS, this function may produce less subgraphs but some of them
 * may be not fully connected(connectivity of subgrpahs will not affect the inference procedure of
 * subgraphs) This function specifically handles the partitioning logic for NPU support and
 * preferred operations.
 *
 * @param     [in] g The original ONNX graph to be partitioned.
 * @param     [out] otherSubgraphs A vector to store the subgraphs that do not meet the preferred
 * operation criteria.
 * @param     [in] d The device object containing information about supported and preferred
 * operations.
 * @param     [in,out] visited An array to keep track of visited nodes.
 * @param     [in] adjacencyList The adjacency list representing the graph's structure.
 * @param     [in] strategy The partitioning strategy to be used (e.g., SPILTE_CPU_STRUCTURE_FIRST,
 * SPILTE_NPU_STRUCTURE_FIRST).
 *
 * @pre       The graph `g` and `adjacencyList` should be properly initialized.
 * @pre       The `visited` array should be initialized to zero.
 * @pre       The device object `d` should be properly initialized with support and preferred
 * operations.
 *
 * @post      The `otherSubgraphs` vector will contain subgraphs that do not meet the preferred
 * operation criteria.
 * @post      The `visited` array will reflect the nodes that have been visited.
 *
 * @exception None
 *
 * @return    None
 */
void DetermineSubgraphs(std::vector<onnx::GraphProto> &subgraphs, const onnx::GraphProto &g,
                        std::vector<onnx::GraphProto> &otherSubgraphs, Device &d, int *visited,
                        std::vector<GraphAdjacencyNode> &adjacencyList, PartitionStrategy strategy)
{
  float maxSubgraphSize = d._max_subgraph_size;
  std::vector<std::string> supportOp;
  std::vector<std::string> preferOp;
  supportOp = d.getNPUSupportOp();
  preferOp = d.getNPUPreferOp();
  onnx::GraphProto tempGraph;
  int endFlag = 0;
  int nodeCount = 0;
  float tempGraphSize = 0;

  while (!endFlag)
  {
    float nodeSize = CalculateNodeSize(g, nodeCount);

    if (tempGraph.node_size() != 0)
    {
      if ((std::find(supportOp.begin(), supportOp.end(), g.node(nodeCount).op_type()) !=
           supportOp.end()) &&
          tempGraph.node_size() <= maxSubgraphSize)
      {
        *tempGraph.add_node() = g.node(nodeCount);
        tempGraphSize += nodeSize;

        if (tempGraphSize > maxSubgraphSize)
        {
          std::cout << "graph size exceed max size!" << tempGraphSize << " " << maxSubgraphSize
                    << std::endl;
        }

        visited[nodeCount] = 1;
      }
      else
      {
        int find_preferop = 0;

        for (int j = 0; j < tempGraph.node_size(); j++)
        {
          if (std::find(preferOp.begin(), preferOp.end(), tempGraph.node(j).op_type()) !=
              preferOp.end())
          {
            find_preferop = 1;
            break;
          }
        }

        if (find_preferop == 1)
        {
          subgraphs.push_back(tempGraph);
        }
        else
        {
          for (int k = 1; k <= tempGraph.node_size(); k++)
          {
            visited[nodeCount - k] = 0;
          }
        }

        tempGraph.Clear();
        tempGraphSize = 0;
        if (std::find(supportOp.begin(), supportOp.end(), g.node(nodeCount).op_type()) !=
            supportOp.end())
        {
          *tempGraph.add_node() = g.node(nodeCount);
          tempGraphSize += nodeSize;
          visited[nodeCount] = 1;
          continue;
        }
      }
    }
    else
    {
      if (std::find(supportOp.begin(), supportOp.end(), g.node(nodeCount).op_type()) !=
          supportOp.end())
      {
        *tempGraph.add_node() = g.node(nodeCount);
        tempGraphSize += nodeSize;

        if (tempGraphSize > maxSubgraphSize)
        {
          std::cout << "graph size exceed max size!" << tempGraphSize << " " << maxSubgraphSize
                    << std::endl;
        }

        visited[nodeCount] = 1;
      }
    }

    nodeCount++;

    if (nodeCount == g.node_size())
    {
      endFlag = 1;

      if (tempGraph.node_size() != 0)
      {
        subgraphs.push_back(tempGraph);
      }
    }
  }

  onnx::GraphProto tempGraph2;
  float tempGraphSize2 = 0;

  for (int i = 0; i < g.node_size(); i++)
  {
    float nodeSize = CalculateNodeSize(g, i);

    if (visited[i] == 0 && tempGraphSize2 < maxSubgraphSize)
    {
      *tempGraph2.add_node() = g.node(i);
      tempGraphSize2 += nodeSize;
    }
    else
    {
      std::cout << "i = " << i << " tempGraphSize2: " << tempGraphSize2 << std::endl;

      if (tempGraph2.node_size() != 0)
      {
        otherSubgraphs.push_back(tempGraph2);
        tempGraphSize2 = 0;
        tempGraph2.Clear();
      }

      if (visited[i] == 0)
      {
        *tempGraph2.add_node() = g.node(i);
        tempGraphSize2 += nodeSize;
        continue;
      }
    }

    if (i == g.node_size() - 1)
    {
      if (tempGraph2.node_size() != 0)
      {
        otherSubgraphs.push_back(tempGraph2);
        tempGraph2.Clear();
      }
    }
  }
}

/**
 * @brief     Perform Tarjan's algorithm to find all strongly connected components in a
 *            directed graph.This function uses depth-first search (DFS) to identify and
 *            group nodes into strongly connected components.
 *
 * @param     [in] index The current node index being visited.
 * @param     [in] depth The current depth in the DFS traversal.
 * @param     [out] stronglyConnectedSubgraphs A vector to store the identified strongly
 *            connected components.
 * @param     [in,out] DFN An array to store the discovery time of each node.
 * @param     [in,out] LOW An array to store the lowest discovery time reachable from each node.
 * @param     [in,out] stackSubgraphs A stack to keep track of nodes in the current DFS path.
 * @param     [in] successorsSubgraphs A vector of vectors representing the adjacency list of
 *            the graph.
 *
 * @pre       The `DFN` and `LOW` arrays should be initialized to zero.
 * @pre       The `stackSubgraphs` should be empty before the first call to this function.
 * @pre       The `successorsSubgraphs` should be properly initialized with the graph's
 *            adjacency list.
 *
 * @post      The `stronglyConnectedSubgraphs` vector will contain all the strongly connected
 *            components found in the graph.
 * @post      The `DFN` and `LOW` arrays will reflect the discovery times and lowest reachable
 *            discovery times for each node.
 * @post      The `stackSubgraphs` will be empty after the function completes.
 *
 * @exception None
 *
 * @return    None
 */
void Tarjan(int index, int depth, std::vector<std::vector<int>> &stronglyConnectedSubgraphs,
            int *DFN, int *LOW, std::vector<int> &stackSubgraphs,
            std::vector<std::vector<int>> &successorsSubgraphs)
{
  int rank = depth + 1;
  DFN[index] = LOW[index] = rank; // initialize DFN and LOW to 0
  stackSubgraphs.push_back(index);

  for (const auto &successor : successorsSubgraphs[index])
  {
    if (DFN[successor] == 0) // the successor is not visited
    {
      Tarjan(successor, rank, stronglyConnectedSubgraphs, DFN, LOW, stackSubgraphs,
             successorsSubgraphs); // visit successor
      LOW[index] = std::min(LOW[index], LOW[successor]);
    }
    else if (std::find(stackSubgraphs.begin(), stackSubgraphs.end(), successor) !=
             stackSubgraphs.end())
    {
      LOW[index] = std::min(LOW[index], DFN[successor]);
    }
  }

  // if this node is the smallest root of the strongly connected component subtree,
  // then subsequent nodes are popped out of the stack and the obtained strongly
  // connected components are saved
  if (LOW[index] == DFN[index])
  {
    auto it = stackSubgraphs.end() - 1;
    std::vector<int> stronglyConnected;

    while (*it != index)
    {
      stronglyConnected.insert(stronglyConnected.begin(), *it);
      stackSubgraphs.pop_back();
      it = stackSubgraphs.end() - 1;
    }

    stronglyConnected.insert(stronglyConnected.begin(), *it);

    if (stronglyConnected.size() > 1)
    {
      stronglyConnectedSubgraphs.push_back(stronglyConnected);
    }

    stackSubgraphs.pop_back(); // pop
  }
}

void PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                    const std::unordered_map<std::string, NodeIOSize> &node_io_size)
{
  return;
}
