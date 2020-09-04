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

#include "luci/Pass/PropagateConcatenationQparamPass.h"

#include "PropagateConcatenationQparamPassInternal.h"

#include <luci/IR/CircleNodes.h>
#include <luci/IR/AttrFusedActFunc.h>
#include <luci/Log.h>

#include <math.h>

namespace
{

bool is_uint8_quantized(luci::CircleNode *node)
{
  if (node->dtype() == loco::DataType::U8 && node->quantparam() != nullptr &&
      node->quantparam()->scale.size() > 0 && node->quantparam()->zerop.size() > 0)
    return true;

  return false;
}

void overwrite_quantparam(luci::CircleConcatenation *concat, luci::CircleNode *target)
{
  auto concat_qparam = concat->quantparam();
  assert(concat_qparam != nullptr);

  auto target_qparam = target->quantparam();
  assert(target_qparam != nullptr);
  target_qparam->min = concat_qparam->min;
  target_qparam->max = concat_qparam->max;
  target_qparam->scale = concat_qparam->scale;
  target_qparam->zerop = concat_qparam->zerop;
  target_qparam->quantized_dimension = concat_qparam->quantized_dimension;
}

void quantize_const(luci::CircleConst *const_node, float scaling_factor, float zerop)
{
  uint32_t size = const_node->size<loco::DataType::FLOAT32>();

  const float scaling_factor_inv = 1.0 / scaling_factor;
  std::vector<int32_t> quantized_values(size);
  for (uint32_t i = 0; i < size; ++i)
  {
    auto data = const_node->at<loco::DataType::FLOAT32>(i);
    quantized_values[i] = static_cast<int32_t>(std::round(data * scaling_factor_inv) + zerop);
  }

  const_node->dtype(loco::DataType::U8);      // change the type of tensor
  const_node->size<loco::DataType::U8>(size); // resize tensor
  for (uint32_t i = 0; i < size; ++i)
  {
    const_node->at<loco::DataType::U8>(i) = std::min(255, std::max(0, quantized_values[i]));
  }
}

} // namespace

namespace luci
{

void propagate_concat_quantparam(luci::CircleConcatenation *concat)
{
  // Check if concat is uint8-quantized
  if (!is_uint8_quantized(concat))
    return;

  // Check if concat has no fused activation function
  if (concat->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return;

  const auto num_inputs = concat->numValues();
  for (uint32_t i = 0; i < num_inputs; i++)
  {
    auto node = loco::must_cast<luci::CircleNode *>(concat->arg(i));

    // Skip if this input is CONCAT Op
    if (node->opcode() == luci::CircleOpcode::CONCATENATION)
      continue;

    // Skip if this input is used by other Ops
    auto succs = loco::succs(node);
    if (succs.size() != 1)
      continue;

    assert(succs.find(concat) != succs.end());

    // Quantize constant values
    if (node->opcode() == luci::CircleOpcode::CIRCLECONST)
    {
      luci::CircleConst *const_node = loco::must_cast<luci::CircleConst *>(node);
      assert(const_node->dtype() == loco::DataType::FLOAT32);
      if (const_node->dtype() != loco::DataType::FLOAT32)
        throw std::runtime_error("Unsupported data type for constant input of concatenation Op");

      const auto concat_qparam = concat->quantparam();
      assert(concat_qparam->scale.size() == 1);
      const auto scaling_factor = concat_qparam->scale[0];
      const auto zerop = concat_qparam->zerop[0];

      quantize_const(const_node, scaling_factor, zerop);
    }
    else
    {
      // Non-const input should be uint8-quantized
      assert(is_uint8_quantized(node));
    }

    overwrite_quantparam(concat, node);
  }
}

/** BEFORE
 *
 *         [CircleNode]             [CircleNode]
 *           (qparam1)                (qparam2)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                          (qparam3)
 *
 *  AFTER
 *         [CircleNode]             [CircleNode]
 *           (qparam3)                (qparam3)
 *                   \                    /
 *                    \                  /
 *                    [CircleConcatenation]
 *                          (qparam3)
 */
bool PropagateConcatenationQparamPass::run(loco::Graph *g)
{
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto concat = dynamic_cast<luci::CircleConcatenation *>(node);
    if (not concat)
      continue;

    // Propagate qparam of concat to its inputs if
    // (1) concat is uint8-quantized
    // (2) concat has no fused activation function
    // (3) the input is not concatenation Op
    // (4) the input is not produced to Ops other than concat
    propagate_concat_quantparam(concat);
  }

  return false;
}

} // namespace luci
