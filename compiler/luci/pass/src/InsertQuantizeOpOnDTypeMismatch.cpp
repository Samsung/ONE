/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "InsertQuantizeOpOnDTypeMismatch.h"
#include "QuantizationUtils.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/Log.h>

#include <limits> // std::numeric_limits

using namespace luci;

namespace
{

// Update u8 node to i16
// Qparam of i16 is inferred from the qparam of u8
void update_u8_to_i16(luci::CircleNode *node)
{
  assert(node->dtype() == loco::DataType::U8); // FIX_CALLER_UNLESS

  node->dtype(loco::DataType::S16);

  auto qparam = node->quantparam();
  assert(qparam);
  assert(qparam->scale.size() == 1);
  assert(qparam->zerop.size() == 1);

  auto u8_scale = qparam->scale[0];
  auto u8_zerop = qparam->zerop[0];

  auto min = u8_scale * (-u8_zerop);
  auto max = u8_scale * (255 - u8_zerop);

  float s16_scale{0};
  int64_t s16_zerop{0};
  float nudged_min{0};
  float nudged_max{0};

  compute_sym_scale(min, max, s16_scale, nudged_min, nudged_max);

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(s16_scale);
  quantparam->zerop.push_back(s16_zerop);

  node->quantparam(std::move(quantparam));
}

// Update i16 node to u8 node
// Qparam of u8 is inferred from the qparam of i16
void update_i16_to_u8(luci::CircleNode *node)
{
  assert(node->dtype() == loco::DataType::S16); // FIX_CALLER_UNLESS

  node->dtype(loco::DataType::U8);

  auto qparam = node->quantparam();
  assert(qparam);
  assert(qparam->scale.size() == 1);
  assert(qparam->zerop.size() == 1);

  auto s16_scale = qparam->scale[0];
  assert(qparam->zerop[0] == 0);

  auto max = s16_scale * std::numeric_limits<int16_t>::max();
  auto min = -max;

  float u8_scale{0};
  int64_t u8_zerop{0};
  float nudged_min{0};
  float nudged_max{0};

  compute_asym_scale_zp(min, max, u8_scale, u8_zerop, nudged_min, nudged_max);

  auto quantparam = std::make_unique<CircleQuantParam>();
  quantparam->scale.push_back(u8_scale);
  quantparam->zerop.push_back(u8_zerop);

  node->quantparam(std::move(quantparam));
}

// Create a Quantize Op which has the same
// dtype, shape, and qparam with node
luci::CircleQuantize *create_quantize_op(luci::CircleNode *node)
{
  auto quantize = node->graph()->nodes()->create<CircleQuantize>();
  quantize->name(node->name() + "_Quantize");
  quantize->dtype(node->dtype());
  quantize->rank(node->rank());
  for (uint32_t i = 0; i < node->rank(); i++)
    quantize->dim(i).set(node->dim(i).value());

  quantize->shape_status(luci::ShapeStatus::VALID);

  assert(node->quantparam()); // FIX_CALLER_UNLESS
  copy_quantparam(node, quantize);

  luci::add_origin(quantize, luci::get_origin(node));

  return quantize;
}

} // namespace

namespace luci
{

void InsertQuantizeOpOnDTypeMismatch::visit(luci::CircleFullyConnected *node)
{
  auto input = loco::must_cast<luci::CircleNode *>(node->input());

  // Input dtype == Output dtype. No problem
  if (input->dtype() == node->dtype())
    return;

  // Skip if node has bias
  if (dynamic_cast<luci::CircleOutputExclude *>(node->bias()) == nullptr)
    return;

  if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return;

  // Only cares quantized case
  if (not is_quantized(input))
    return;

  if (not is_quantized(node))
    return;

  // Let's support limited case
  // TODO Extend this to another dtype
  if (input->dtype() != loco::DataType::U8)
    return;

  if (node->dtype() != loco::DataType::S16)
    return;

  // Create Quantize Op
  auto quant_op = create_quantize_op(node);

  // Insert Quantize Op after node
  loco::replace(node).with(quant_op);
  quant_op->input(node);

  // Update node's dtype and qparam from i16 to u8
  // NOTE This would severely degrade accuracy. It is
  // important to mitigate this accuracy drop in backend.
  update_i16_to_u8(node);
}

void InsertQuantizeOpOnDTypeMismatch::visit(luci::CircleMul *node)
{
  auto x = loco::must_cast<luci::CircleNode *>(node->x());
  auto y = loco::must_cast<luci::CircleNode *>(node->y());

  assert(x->dtype() == y->dtype()); // FIX_CALLER_UNLESS

  // Ignore invalid dtype
  if (x->dtype() != y->dtype())
    return;

  if (node->fusedActivationFunction() != luci::FusedActFunc::NONE)
    return;

  // Input dtype == Output dtype. No problem
  if (x->dtype() == node->dtype())
    return;

  // Only cares quantized case
  if (not is_quantized(x))
    return;

  if (not is_quantized(y))
    return;

  if (not is_quantized(node))
    return;

  // Let's support limited case
  // TODO Extend this to another dtype
  if (x->dtype() != loco::DataType::S16)
    return;

  if (node->dtype() != loco::DataType::U8)
    return;

  // Create Quantize Op
  auto quant_op = create_quantize_op(node);

  // Insert Quantize Op after node
  loco::replace(node).with(quant_op);
  quant_op->input(node);

  // Update node's dtype and qparam from u8 to i16
  update_u8_to_i16(node);
}

void InsertQuantizeOpOnDTypeMismatch::visit(luci::CircleBatchMatMul *node)
{
  auto x = loco::must_cast<luci::CircleNode *>(node->x());
  auto y = loco::must_cast<luci::CircleNode *>(node->y());

  assert(x->dtype() == y->dtype()); // FIX_CALLER_UNLESS

  // Ignore invalid dtype
  if (x->dtype() != y->dtype())
    return;

  if (node->adj_x() or node->adj_y())
    return;

  // Input dtype == Output dtype. No problem
  if (x->dtype() == node->dtype())
    return;

  // Only cares quantized case
  if (not is_quantized(x))
    return;

  if (not is_quantized(y))
    return;

  if (not is_quantized(node))
    return;

  // Let's support limited case
  // TODO Extend this to another dtype
  if (x->dtype() != loco::DataType::S16)
    return;

  if (node->dtype() != loco::DataType::U8)
    return;

  // Create Quantize Op
  auto quant_op = create_quantize_op(node);

  // Insert Quantize Op after node
  loco::replace(node).with(quant_op);
  quant_op->input(node);

  // Update node's dtype and qparam from i16 to u8
  update_u8_to_i16(node);
}

} // namespace luci
