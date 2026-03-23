/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Pass/Passes/SqueezeReduceNode.h"

#include <moco/Support/NodeAs.h>

#include <moco/IR/Nodes/TFConst.h>
#include <moco/IR/Nodes/TFSqueeze.h>
#include <moco/IR/Nodes/TFMean.h>

#include <cassert>

namespace
{

/**
 * WHEN:
 *   - Reduce operations do not keep dimensions
 * DO:
 *   - Replace original ReduceTypeOp to new ReduceTypeOp, which 'keep_dims' attribute is true
 *   - Insert TFSqueeze after new ReduceTypeOp
 *
 *
 * <Before>
 *     in ---- ReduceTypeOp:0 (keep_dims = false) --- out(s)
 *
 * <After>
 *         --- ReduceTypeOp:0 (keep_dims = false)
 *        /
 *     in ---- ReduceTypeOp:1 (keep_dims = true) ---- TFSqueeze --- out(s)
 *
 * <Where>
 *  - 'keep_dims' attribute of ReduceTypeOp:0 is false
 *
 */
template <class TFNode> bool squeeze_reduce_node(loco::Graph *graph, TFNode *reduce_node)
{
  // Don't need to squeeze reduce node
  if (reduce_node->keep_dims())
    return false;

  // Reduction indices are not yet constant
  auto const_reduction_indices = dynamic_cast<moco::TFConst *>(reduce_node->reduction_indices());
  if (const_reduction_indices == nullptr)
    return false;

  auto squeeze_node = graph->nodes()->create<moco::TFSqueeze>();
  auto new_reduce_node = graph->nodes()->create<TFNode>();

  new_reduce_node->input(reduce_node->input());
  new_reduce_node->reduction_indices(reduce_node->reduction_indices());
  new_reduce_node->keep_dims(true);

  // Insert squeeze dims
  // TODO Support S64 type
  assert(const_reduction_indices->dtype() == loco::DataType::S32);

  std::vector<int64_t> reduction_values;
  for (uint32_t i = 0; i < const_reduction_indices->size<loco::DataType::S32>(); ++i)
    reduction_values.push_back(const_reduction_indices->at<loco::DataType::S32>(i));
  squeeze_node->squeeze_dims(reduction_values);

  // replace
  loco::replace(reduce_node).with(squeeze_node);
  squeeze_node->input(new_reduce_node);

  return true;
}

} // namespace

namespace moco
{

bool SqueezeReduceNode::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto shape_node = as<moco::TFMean>(node))
    {
      if (squeeze_reduce_node(graph, shape_node))
        changed = true;
    }
    // TODO Add more reduce type operations
  }

  return changed;
}

} // namespace moco
