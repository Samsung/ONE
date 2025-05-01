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
 * @param Subgraphs A vector containing subgraph information.
 * @param SubgraphFileName The filename for the output of subgraph information.
 * @param OtherSubgraphs A vector containing other subgraph information.
 * @param OtherSubgraphFileName The filename for the output of other subgraph information.
 */
void PrintSubgraphs(std::vector<onnx::GraphProto> Subgraphs, char *SubgraphFileName,
                    std::vector<onnx::GraphProto> OtherSubgraphs, char *OtherSubgraphFileName)
{
  int nodeSum = 0;

  std::ofstream outFile(SubgraphFileName);
  if (!outFile.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  int id = 0;
  for (const auto &vec : Subgraphs)
  {
    outFile << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFile << node.name() << " ";
    }

    id++;
    outFile << std::endl;
    nodeSum += vec.node_size();
  }

  std::ofstream outFileOther(OtherSubgraphFileName);
  if (!outFileOther.is_open())
  {
    std::cerr << "Error opening file." << std::endl;
    exit(-1);
  }

  std::cout << "before:" << std::endl;
  for (const auto &vec : OtherSubgraphs)
  {
    outFileOther << " subgraph" << id << ":";
    for (const auto &node : vec.node())
    {
      outFileOther << node.name() << " ";
    }

    id++;
    outFileOther << std::endl;
    nodeSum += vec.node_size();
  }
}

void PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                    const std::unordered_map<std::string, NodeIOSize> &node_io_size)
{
  return;
}
