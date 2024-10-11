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

#include "QuantizeWeightsLLM.h"

#include "QuantizeUtil.h"

#include <luci/Service/Nodes/CircleConst.h>

namespace
{

bool is_quantized(const luci::CircleConst *node)
{
  return node->quantparam() != nullptr || (node->dtype() != loco::DataType::FLOAT32);
}

size_t elementsize(const luci::CircleConst *node)
{
  size_t elems = 1;
  for (uint32_t i = 0; i < node->rank(); ++i)
    elems *= node->dim(i).value();
  return elems;
}

luci::CircleConst *quantize_q8_block(luci::CircleConst *node)
{
  auto new_weights = luci::clone(node);

  // Check block size
  auto last_dim = node->dim(node->rank() - 1).value();
  assert(last_dim % QK8_0 == 0);

  // Get num of block
  size_t blocks = 1;
  for (uint32_t i = 0; i < new_weights->rank(); ++i)
    blocks *= new_weights->dim(i).value();
  blocks /= QK8_0;

  new_weights->dtype(loco::DataType::GGML_Q8_0);

  // Set data for each block
  block_q8_0_u block;
  // Fake data type for resize and write
  new_weights->size<loco::DataType::U8>(sizeof(block) * blocks);
  for (size_t i = 0; i < blocks; ++i)
  {
    // Read float data
    float data[QK8_0];
    for (size_t j = 0; j < QK8_0; ++j)
      data[j] = node->at<loco::DataType::FLOAT32>(i * QK8_0 + j);

    ggml_quantize_q8_0(data, &block.b, QK8_0, QK8_0);

    for (auto j = 0; j < sizeof(block.u8); j++)
      new_weights->at<loco::DataType::U8>(i * sizeof(block.u8) + j) = block.u8[j];
  }

  new_weights->dtype(loco::DataType::GGML_Q8_0);
  return new_weights;
}

luci::CircleConst *quantize_q4_block(luci::CircleConst *node)
{
  auto new_weights = luci::clone(node);

  // Check block size
  auto last_dim = node->dim(node->rank() - 1).value();
  assert(last_dim % QK4_0 == 0);

  // Get num of block
  size_t blocks = 1;
  for (uint32_t i = 0; i < new_weights->rank(); ++i)
    blocks *= new_weights->dim(i).value();
  blocks /= QK4_0;

  new_weights->dtype(loco::DataType::GGML_Q4_0);

  // Set data for each block
  block_q4_0_u block;
  // Fake data type for resize and write
  new_weights->size<loco::DataType::U8>(sizeof(block) * blocks);
  for (size_t i = 0; i < blocks; ++i)
  {
    // Read float data
    float data[QK4_0];
    for (size_t j = 0; j < QK4_0; ++j)
      data[j] = node->at<loco::DataType::FLOAT32>(i * QK4_0 + j);

    ggml_quantize_q4_0(data, &block.b, QK4_0, QK4_0);

    for (auto j = 0; j < sizeof(block.u8); j++)
      new_weights->at<loco::DataType::U8>(i * sizeof(block.u8) + j) = block.u8[j];
  }

  return new_weights;
}

} // namespace

namespace quantizer
{

void QuantizeWeightsLLM::visit(luci::CircleFullyConnected *node)
{
  auto weights = loco::must_cast<luci::CircleConst *>(node->weights());
  if (elementsize(weights) < _skip_length)
    return;

  if (!is_quantized(weights))
  {
    auto new_weights =
      _quant_type == Type::Q4_0 ? quantize_q4_block(weights) : quantize_q8_block(weights);
    node->weights(new_weights);
  }
}

void QuantizeWeightsLLM::visit(luci::CircleGather *node)
{
  if (dynamic_cast<luci::CircleConst *>(node->params()) == nullptr)
    return;

  if (dynamic_cast<luci::CircleConst *>(node->indices()) != nullptr)
    return;

  auto input = loco::must_cast<luci::CircleConst *>(node->arg(0));
  if (elementsize(input) < _skip_length)
    return;

  if (!is_quantized(input))
  {
    // Workaround: indices to INT32 type
    auto indices = loco::must_cast<luci::CircleNode *>(node->indices());

    if (_quant_type == Type::SKIP)
    {
      indices->dtype(loco::DataType::S32);
      return;
    }

    auto new_weights =
      _quant_type == Type::Q4_0 ? quantize_q4_block(input) : quantize_q8_block(input);
    node->params(new_weights);

    if (indices->dtype() == loco::DataType::S64)
      indices->dtype(loco::DataType::S32);
  }
}

void QuantizeWeightsLLM::visit(luci::CircleNode *) {}

} // namespace quantizer
