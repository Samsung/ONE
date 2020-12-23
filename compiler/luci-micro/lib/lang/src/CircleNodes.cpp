/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/IR/CircleNodes.h"

#include "Check.h"

#include <loco.h>

namespace luci
{

void set_new_shape(CircleReshape *node, int32_t *base, uint32_t size)
{
  // Check node does not have both of new shape infos
  LUCI_ASSERT(node->shape() == nullptr, "node already has shape input");
  LUCI_ASSERT(node->newShape()->rank() == 0, "node already has newShape attribute");

  const loco::DataType S32 = loco::DataType::S32;

  // Set 2nd input as CircleConst
  auto const_shape_node = node->graph()->nodes()->create<CircleConst>();
  const_shape_node->rank(1);
  const_shape_node->dim(0) = size;
  const_shape_node->dtype(S32);
  const_shape_node->size<S32>(size);
  for (uint32_t axis = 0; axis < size; ++axis)
    const_shape_node->at<S32>(axis) = base[axis];
  node->shape(const_shape_node);

  // Set newShape attribute
  node->newShape()->rank(size);
  for (uint32_t axis = 0; axis < size; ++axis)
    node->newShape()->dim(axis) = base[axis];
}

void link(loco::GraphOutput *output, CircleOutput *node) { node->index(output->index()); }

CircleOutput *output_node(loco::Graph *g, const loco::GraphOutputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto output = dynamic_cast<CircleOutput *>(g->nodes()->at(n)))
    {
      if (output->indexed() && output->index() == index)
      {
        return output;
      }
    }
  }
  return nullptr;
}

void link(loco::GraphInput *input, CircleInput *node) { node->index(input->index()); }

CircleInput *input_node(loco::Graph *g, const loco::GraphInputIndex &index)
{
  for (uint32_t n = 0; n < g->nodes()->size(); ++n)
  {
    if (auto input = dynamic_cast<CircleInput *>(g->nodes()->at(n)))
    {
      if (input->indexed() && input->index() == index)
      {
        return input;
      }
    }
  }
  return nullptr;
}

} // namespace luci
