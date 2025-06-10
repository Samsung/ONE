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

#ifndef __TOOLS_ONNX_SUBGRAPH_PARTITION_H__
#define __TOOLS_ONNX_SUBGRAPH_PARTITION_H__

#include "onnx.pb.h"
#include "device.h"
#include "graph.h"

// Priority setting when OPs can be supported by both CPU & NPU, and performance is same on both
enum PartitionStrategy
{
  SPILTE_CPU_STRUCTURE_FIRST,
  SPILTE_NPU_STRUCTURE_FIRST,
  AUTOMATIC_SEARCH
};

/**
 * @brief     Partition the ONNX graph into subgraphs and produce cutting instructions.
 *
 * @param     [in] g The ONNX graph to be partitioned.
 * @param     [in] d The device information for partitioning.
 * @param     [in] strategy The partition strategy to be used (deprecated).
 * @param     [in] node_io_size The input/output size information for each node.
 * @pre       The ONNX graph should be valid and the device information should be properly set.
 * @post      The graph is partitioned into subgraphs, and the results are stored in Subgraphs and
 * otherSubgraphs.
 * @exception None
 * @return    None
 */
void PartitionGraph(const onnx::GraphProto &g, Device &d, PartitionStrategy strategy,
                    const std::unordered_map<std::string, NodeIOSize> &node_io_size);

#endif // __TOOLS_ONNX_SUBGRAPH_PARTITION_H__
