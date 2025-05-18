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

void PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                    const std::unordered_map<std::string, NodeIOSize> &node_io_size)
{
  return;
}
