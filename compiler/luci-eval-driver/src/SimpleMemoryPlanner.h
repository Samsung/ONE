/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_SIMPLE_MEMORY_PLANNER_H
#define LUCI_INTERPRETER_SIMPLE_MEMORY_PLANNER_H

#include <luci/IR/Module.h>
#include "luci_interpreter/CircleNodeMemoryPlan.h"
#include "luci/Profile/CircleNodeID.h"

namespace luci
{

struct AllocNodeInform {

  AllocNodeInform() {
    offset = 0;
    size = 0;
    node_num = -1;
    first_node = -1;
    last_node = -1;
    is_temp = false;
    breadth = 0;
  }

  uint32_t offset;
  uint32_t size;
  uint32_t node_num;
  uint32_t first_node;
  uint32_t last_node;
  bool is_temp;
  uint32_t breadth;

  bool operator<(const AllocNodeInform& other) const {
    return offset < other.offset;
  }
};

class SimpleMemoryPlanner
{
public:
  SimpleMemoryPlanner() = delete;
  SimpleMemoryPlanner(luci::Module *module)
  {
    _graph = module->graph();
  };

  // Method plans memory allocations for nodes.
  // And then write in nodes annotation information about necessary offsets in arena memory buffer.
  uint32_t PlanAllocations();

private:
  // Method gets nodes usage information.
  void PlanOrder();
  // Method gets information about sizes of different type of nodes.
  void GetSizesInformation(uint32_t required_size);
  // Method dump information about nodes required sizes.
  void DumpInform();

  // Method finds offset for nodes in some buffer.
  uint32_t PlanOffset();
  // Method finds size of buffer for naive approach.
  uint32_t NaiveSize();
  // Method realizes greedy_by_size approach.
  uint32_t GreedyBySize();
  // Method finds (if necessary) size for im2col temporary tensor.
  uint32_t Im2colSize(const luci::CircleConv2D *conv);
  // Method creates and fills _alloc_inform_vector with usage interval inform and node's sizes.
  void CreateAllocVector(bool null_consts = false,
                         bool null_inputs = false,
                         bool null_im2col = false);

  // Stores allocation data for all tensors.
  std::vector<AllocNodeInform> _alloc_inform_vector;
  // Stores nodes in execution order.
  std::vector<loco::Node *> _ordered_nodes;
  // Stores nodes offsets in arena buffer
  std::vector<std::vector<uint32_t>> _offsets;
  // Stores position (in _ordered_nodes vector) of node,
  // where node in i'th position in this vector first use
  std::vector<uint32_t> _alloc_node;
  // Stores position (in _ordered_nodes vector) of node,
  // where node in i'th position in this vector last use
  std::vector<uint32_t> _dealloc_node;

  loco::Graph * _graph;
  uint32_t _required_size = 0;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_SIMPLE_MEMORY_PLANNER_H