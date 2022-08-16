/*
 * Copyright (c) 2022 Samsung Electronics Co., Ltd. All Rights Reserved
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

#include "luci/Pass/FoldDensifyPass.h"
#include "helpers/SparsityFormatConverter.h"

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <cassert>
#include <vector>

namespace
{

bool is_foldable_const(luci::CircleConst *node)
{
  if (node->sparsityparam() == nullptr)
    return false;

  if (node->dtype() == loco::DataType::FLOAT32)
    return true;
  if (node->dtype() == loco::DataType::FLOAT16)
    return true;

  return false;
}

luci::CircleConst *densified_const_node(luci::CircleConst *const_node)
{
  assert(const_node->sparsityparam());

  auto name = const_node->name();
  assert(name.length() > 0);
  auto g = const_node->graph();
  auto new_const_node = g->nodes()->create<luci::CircleConst>();

  new_const_node->dtype(const_node->dtype());
  new_const_node->rank(const_node->rank());

  uint32_t dim_size = 1;
  std::vector<int> dense_shape;
  for (uint32_t i = 0; i < new_const_node->rank(); ++i)
  {
    assert(const_node->dim(i).known());
    new_const_node->dim(i) = const_node->dim(i);

    uint32_t value = const_node->dim(i).value();
    dim_size *= value;
    dense_shape.emplace_back(static_cast<int32_t>(value));
  }

  if (const_node->dtype() == loco::DataType::FLOAT32)
    new_const_node->size<loco::DataType::FLOAT32>(dim_size);
  else
  {
    assert(const_node->dtype() == loco::DataType::FLOAT16);
    new_const_node->size<loco::DataType::FLOAT16>(dim_size);
  }

  new_const_node->shape_status(luci::ShapeStatus::VALID);
  new_const_node->name(name + "_DS");

  if (const_node->dtype() == loco::DataType::FLOAT32)
  {
    auto const_items = const_node->size<loco::DataType::FLOAT32>();
    auto f_data = std::make_unique<float[]>(const_items);
    for (size_t i = 0; i < const_items; ++i)
      f_data[i] = const_node->at<loco::DataType::FLOAT32>(i);

    sparsity::TfLiteSparsity sp = to_tflite_sparsity(const_node->sparsityparam());
    sparsity::FormatConverter<float> converter(dense_shape, sp);
    converter.SparseToDense(f_data.get());
    const auto &data_dense = converter.GetData();
    assert(data_dense.size() == dim_size);

    for (uint32_t i = 0; i < dim_size; ++i)
      new_const_node->at<loco::DataType::FLOAT32>(i) = data_dense[i];

    luci::freeTfLiteSparsity(sp);
  }
  else
  {
    assert(const_node->dtype() == loco::DataType::FLOAT16);

    auto const_items = const_node->size<loco::DataType::FLOAT16>();
    auto f_data = std::make_unique<uint16_t[]>(const_items);
    for (size_t i = 0; i < const_items; ++i)
      f_data[i] = const_node->at<loco::DataType::FLOAT16>(i);

    // Primitive type for FLOAT16 is UINT16
    sparsity::TfLiteSparsity sp = to_tflite_sparsity(const_node->sparsityparam());
    sparsity::FormatConverter<uint16_t> converter(dense_shape, sp);
    converter.SparseToDense(f_data.get());
    const auto &data_dense = converter.GetData();
    assert(data_dense.size() == dim_size);
    for (uint32_t i = 0; i < dim_size; ++i)
      new_const_node->at<loco::DataType::FLOAT16>(i) = data_dense[i];

    luci::freeTfLiteSparsity(sp);
  }

  return new_const_node;
}

/**
 * @brief Fold Densify if input is Sparse Constant
 */
bool fold_densify(luci::CircleDensify *densify)
{
  auto const_input = dynamic_cast<luci::CircleConst *>(densify->input());
  if (not const_input)
    return false;

  if (not is_foldable_const(const_input))
    return false;

  auto dense_const = densified_const_node(const_input);
  assert(dense_const);

  loco::replace(densify).with(dense_const);
  luci::add_origin(dense_const, luci::composite_origin(
                                  {luci::get_origin(densify), luci::get_origin(const_input)}));

  return true;
}

} // namespace

namespace luci
{

/**
 * BEFORE
 *
 *    [CircleConst](sparse)
 *         |
 *   [CircleDensify]
 *         |
 *    [CircleNode]
 *         |
 *
 * AFTER
 *
 *    [CircleConst](dense)  [CircleConst](sparse)
 *         |                     |
 *    [CircleNode]          [CircleDensify]
 *         |
 */
bool FoldDensifyPass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto densify = dynamic_cast<luci::CircleDensify *>(node))
    {
      if (fold_densify(densify))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
