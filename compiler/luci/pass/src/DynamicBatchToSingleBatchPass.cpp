/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/DynamicBatchToSingleBatchPass.h"

#include <luci/IR/CircleNode.h>
#include <loco.h>

namespace luci
{

bool DynamicBatchToSingleBatchPass::run(loco::Graph *g)
{
  assert(g); // FIX CALLER UNLESS

  bool changed = false;

  auto graph_inputs = g->inputs();

  // Assume the first dimension is batch dimension
  const uint32_t BATCH_DIM = 0;

  for (auto node : loco::input_nodes(g))
  {
    auto input_node = loco::must_cast<luci::CircleInput *>(node);

    if (input_node->rank() == 0)
      continue;

    // Skip if batch dimension is known
    if (input_node->dim(BATCH_DIM).known())
      continue;

    if (input_node->rank() != 4)
    {
      // Limit use only for rank 4 inputs (for NHWC and NCHW)
      // TODO Enable this if necessary
      throw std::runtime_error("First dimension of input is unknown, but its rank is not 4.");
    }

    // 'set' will make the dimension known
    input_node->dim(BATCH_DIM).set(1);

    // Update graph input
    auto graph_input = graph_inputs->at(input_node->index());
    auto graph_input_shape = graph_input->shape();
    auto tensor_shape = std::make_unique<loco::TensorShape>();
    {
      tensor_shape->rank(graph_input_shape->rank());
      for (uint32_t i = 0; i < tensor_shape->rank(); i++)
      {
        tensor_shape->dim(i) = graph_input_shape->dim(i);
      }
      tensor_shape->dim(BATCH_DIM).set(1);
    }

    graph_input->shape(std::move(tensor_shape));

    changed = true;
  }

  return changed;
}

} // namespace luci
