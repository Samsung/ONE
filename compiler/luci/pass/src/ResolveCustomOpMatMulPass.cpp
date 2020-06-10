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

#include "luci/Pass/ResolveCustomOpMatMulPass.h"

#include "flatbuffers/flexbuffers.h"
#include <loco/IR/DataTypeTraits.h>

#include <luci/IR/CircleNodes.h>

#include <loco.h>
#include <oops/InternalExn.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/TypeInference.h>

namespace
{

template <typename T>
luci::CircleConst *create_const_node(loco::Graph *g, const loco::DataType dtype,
                                     const std::vector<uint32_t> &shape,
                                     const std::vector<T> &values)
{
  auto node = g->nodes()->create<luci::CircleConst>();
  node->dtype(dtype);
  node->rank(shape.size());
  uint32_t size = (shape.size() == 0) ? 0 : 1;

  for (uint32_t i = 0; i < shape.size(); ++i)
  {
    node->dim(i) = shape.at(i);
    size *= shape.at(i);
  }

  switch (dtype)
  {
    case loco::DataType::S32:
      node->size<loco::DataType::S32>(size);
      for (uint32_t i = 0; i < values.size(); ++i)
        node->at<loco::DataType::S32>(i) = values[i];
      break;
    case loco::DataType::S64:
      node->size<loco::DataType::S64>(size);
      for (uint32_t i = 0; i < values.size(); ++i)
        node->at<loco::DataType::S64>(i) = values[i];
      break;
    case loco::DataType::FLOAT32:
      node->size<loco::DataType::FLOAT32>(size);
      for (uint32_t i = 0; i < values.size(); ++i)
        node->at<loco::DataType::FLOAT32>(i) = values[i];
      break;
    default:
      break;
  }
  return node;
}

bool resolve_matmul(luci::CircleCustom *cop)
{
  auto graph = cop->graph();
  const std::vector<uint8_t> custom_options = cop->custom_options();
  auto map = flexbuffers::GetRoot(custom_options).AsMap();
  const auto S32 = loco::DataType::S32;

  bool transpose_a = map["transpose_a"].AsBool();
  bool transpose_b = map["transpose_b"].AsBool();

  loco::Node *lhs = cop->inputs(0);
  loco::Node *rhs = cop->inputs(1);

  if (transpose_a)
  {
    if (!loco::shape_known(lhs))
      return false;

    auto a_shape = loco::shape_get(lhs).as<loco::TensorShape>();

    if (a_shape.rank() < 2)
      return false;

    // Create a permutation constant node
    std::vector<uint32_t> perm;
    for (uint32_t i = 0; i < a_shape.rank(); ++i)
      perm.push_back(i);
    std::swap(perm[a_shape.rank() - 1], perm[a_shape.rank() - 2]);
    auto perm_node = create_const_node(graph, S32, {a_shape.rank()}, perm);

    // Now make a transpose node
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(lhs);
    transpose_node->perm(perm_node);
    lhs = transpose_node;
  }

  auto b_shape = loco::shape_get(rhs).as<loco::TensorShape>();
  if (b_shape.rank() != 2)
    return false;

  // Transpose the second input if needed. TFLite FullyConnected operator
  // assumes the second input is in column-major order, but the input is
  // in row-major order, thus we need to convert between them.
  if (!transpose_b)
  {
    const std::vector<uint32_t> perm{1, 0};
    auto perm_node = create_const_node(graph, S32, {2}, perm);
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(rhs);
    transpose_node->perm(perm_node);
    rhs = transpose_node;
  }

  // Make a constant zero-filled bias node
  const std::vector<float> val{};
  auto bias_node = create_const_node(graph, loco::dtype_get(cop->inputs(0)),
                                     {b_shape.dim(transpose_b ? 1 : 0).value()}, val);

  auto fc_node = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_node->input(lhs);
  fc_node->weights(rhs);
  fc_node->bias(bias_node);
  fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);

  replace(cop).with(fc_node);
  return true;
}

} // namespace

namespace luci
{

bool ResolveCustomOpMatMulPass::run(loco::Graph *g)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    auto cop = dynamic_cast<luci::CircleCustom *>(node);
    if (not cop)
      continue;

    if (cop->custom_code() != "MatMul")
      continue;

    if (!resolve_matmul(cop))
      continue;

    changed = true;
  }

  return changed;
}

} // namespace luci
