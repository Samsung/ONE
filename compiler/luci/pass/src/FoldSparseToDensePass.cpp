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

#include "luci/Pass/FoldSparseToDensePass.h"

#include <luci/IR/CircleNodes.h>

#include <limits>

namespace
{

/**
 * Fold to const if
 *
 * 1. indices has 0-sized static shape such as [0]
 *    (i.e., output is filled with default value)
 * 2. default_value: const scalar
 * 3. output_shape: const
 *
 * TODO: Support more general patterns
 **/
template <loco::DataType IndexT, loco::DataType ValueT>
bool fold_sparse_to_dense(luci::CircleSparseToDense *stod)
{
  const auto indices = loco::must_cast<luci::CircleNode *>(stod->indices());
  const auto default_value = loco::must_cast<luci::CircleConst *>(stod->default_value());
  const auto output_shape = loco::must_cast<luci::CircleConst *>(stod->output_shape());

  bool has_zero = false;
  for (uint32_t i = 0; i < indices->rank(); i++)
  {
    if (indices->dim(i).known() && indices->dim(i).value() == 0)
      has_zero = true;
  }
  if (!has_zero)
    return false;

  if (default_value->rank() != 0 || default_value->size<ValueT>() != 1)
    return false;

  auto rank = output_shape->size<IndexT>();
  std::vector<uint32_t> shape;
  for (uint32_t i = 0; i < rank; i++)
  {
    auto dim = output_shape->at<IndexT>(i);
    assert(dim >= 0 && dim <= std::numeric_limits<uint32_t>::max());
    if (!(dim >= 0 && dim <= std::numeric_limits<uint32_t>::max()))
      return false;

    shape.push_back(dim);
  }

  auto name = stod->name();
  assert(name.length() > 0);
  auto constant = stod->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(default_value->dtype());
  constant->rank(rank);
  uint32_t dim_size = 1;
  for (uint32_t i = 0; i < rank; i++)
  {
    constant->dim(i).set(shape[i]);
    dim_size *= shape[i];
  }

  constant->size<ValueT>(dim_size);
  const auto value = default_value->scalar<ValueT>();
  for (uint32_t i = 0; i < dim_size; i++)
    constant->at<ValueT>(i) = value;

  constant->shape_status(luci::ShapeStatus::VALID);
  constant->name(name + "_D");

  loco::replace(stod).with(constant);

  return true;
}

bool fold_sparse_to_dense(luci::CircleSparseToDense *stod)
{
  auto indices = loco::must_cast<luci::CircleNode *>(stod->indices());
  auto default_value = dynamic_cast<luci::CircleConst *>(stod->default_value());
  if (not default_value)
    return false;

  auto output_shape = dynamic_cast<luci::CircleConst *>(stod->output_shape());
  if (not output_shape)
    return false;

  // Illegal input check
  if (indices->dtype() != output_shape->dtype())
    throw std::runtime_error("indices and output_shape of SparseToDense must have the same dtype");

  // TODO: Support more data types
  if (indices->dtype() == loco::DataType::S64)
  {
    if (default_value->dtype() == loco::DataType::S64)
    {
      return fold_sparse_to_dense<loco::DataType::S64, loco::DataType::S64>(stod);
    }
  }
  return false;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for SparseToDense Op
 **/
bool FoldSparseToDensePass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto stod = dynamic_cast<luci::CircleSparseToDense *>(node))
    {
      if (fold_sparse_to_dense(stod))
        changed = true;
    }
  }

  return changed;
}

} // namespace luci
