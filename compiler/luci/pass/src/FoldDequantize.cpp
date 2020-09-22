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

#include "luci/Pass/FoldDequantize.h"

#include <luci/IR/CircleNodes.h>

#include <cassert>
#include <string>
#include <set>

namespace
{

bool is_hybrid_supported(loco::Node *node)
{
  if (dynamic_cast<luci::CircleFullyConnected *>(node) != nullptr)
    return true;

  return false;
}
}

namespace luci
{

bool FoldDequantizePass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::all_nodes(g))
  {
    if (auto circle_dequant = dynamic_cast<luci::CircleDequantize *>(node))
    {
      auto input_node = dynamic_cast<luci::CircleNode *>(circle_dequant->input());

      if (input_node->dtype() == loco::DataType::FLOAT32)
      {
        // Input is already dequantized
        loco::replace(circle_dequant).with(input_node);
        changed = true;
      }
    }
    else if (auto const_node = dynamic_cast<luci::CircleConst *>(node))
    {
      if (const_node->dtype() != loco::DataType::U8 && const_node->dtype() != loco::DataType::S8)
        continue;
      if (const_node->quantparam() != nullptr)
      {
        for (auto s : loco::succs(const_node))
        {
          if (!is_hybrid_supported(s))
          {
            // Input can be pre-calculated
            auto circle_output = g->nodes()->create<luci::CircleConst>();

            circle_output->dtype(loco::DataType::FLOAT32);
            circle_output->rank(const_node->rank());
            uint32_t dim_size = 1;
            for (uint32_t i = 0; i < circle_output->rank(); ++i)
            {
              circle_output->dim(i) = const_node->dim(i);
              dim_size *= const_node->dim(i).value();
            }
            circle_output->size<loco::DataType::FLOAT32>(dim_size);
            circle_output->shape_status(luci::ShapeStatus::VALID);

            const int32_t q_dim = const_node->quantparam()->quantized_dimension;
            const int32_t q_dim_value = const_node->dim(q_dim).value();

            int32_t right_count = q_dim_value;
            for (uint32_t i = q_dim + 1; i < const_node->rank(); ++i)
              right_count *= const_node->dim(i).value();

            uint32_t const_node_size = (const_node->dtype() == loco::DataType::S8)
                                           ? const_node->size<loco::DataType::S8>()
                                           : const_node->size<loco::DataType::U8>();
            for (uint32_t i = 0; i < const_node_size; ++i)
            {
              uint32_t qd = (i % right_count) / (right_count / q_dim_value);
              if (qd >= const_node->quantparam()->zerop.size())
                qd = 0;

              if (const_node->dtype() == loco::DataType::S8)
                circle_output->at<loco::DataType::FLOAT32>(i) =
                    ((double)const_node->at<loco::DataType::S8>(i) -
                     const_node->quantparam()->zerop.at(qd)) *
                    const_node->quantparam()->scale.at(qd);
              else if (const_node->dtype() == loco::DataType::U8)
                circle_output->at<loco::DataType::FLOAT32>(i) =
                    ((double)const_node->at<loco::DataType::U8>(i) -
                     const_node->quantparam()->zerop.at(qd)) *
                    const_node->quantparam()->scale.at(qd);
              else
                assert(false && "unsupported fold dequant type");
            }

            loco::replace(const_node).with(circle_output);
            changed = true;
          }
        }
      }
    }
  }

  return changed;
}

} // namespace luci
