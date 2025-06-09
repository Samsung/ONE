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

void PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                    const std::unordered_map<std::string, NodeIOSize> &node_io_size)
{
  return;
}
