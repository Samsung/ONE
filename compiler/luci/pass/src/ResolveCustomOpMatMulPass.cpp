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

#include "helpers/CreateCircleConst.h"

#include <loco/IR/DataTypeTraits.h>

#include <luci/IR/CircleNodes.h>
#include <luci/Profile/CircleNodeOrigin.h>

#include <loco.h>
#include <oops/InternalExn.h>

#include <flatbuffers/flexbuffers.h>

namespace
{

bool resolve_matmul(luci::CircleCustom *cop)
{
#define CHECK_OR_FALSE(condition) \
  if (not(condition))             \
    return false;
#define CHECK_OR_THROW(condition, message) \
  if (not(condition))                      \
    INTERNAL_EXN(message);

  auto graph = cop->graph();
  const std::vector<uint8_t> custom_options = cop->custom_options();
  auto map = flexbuffers::GetRoot(custom_options).AsMap();
  const auto U8 = loco::DataType::U8;
  const auto S16 = loco::DataType::S16;
  const auto S32 = loco::DataType::S32;
  const auto FLOAT32 = loco::DataType::FLOAT32;

  auto name = cop->name();
  assert(name.length() > 0);

  bool transpose_a = map["transpose_a"].AsBool();
  bool transpose_b = map["transpose_b"].AsBool();

  loco::Node *lhs = cop->inputs(0);
  loco::Node *rhs = cop->inputs(1);

  // Check that the type of the first input is known
  auto lhs_dtype = loco::must_cast<luci::CircleNode *>(cop->inputs(0))->dtype();
  CHECK_OR_FALSE(lhs_dtype != loco::DataType::Unknown);

  // If transpose of first input is requested, its shape must be known
  auto circle_lhs = loco::must_cast<luci::CircleNode *>(lhs);
  CHECK_OR_FALSE(!transpose_a || circle_lhs->shape_status() == luci::ShapeStatus::VALID);
  // and its rank should be at least 2
  CHECK_OR_FALSE(!transpose_a || circle_lhs->rank() >= 2);
  // Check that the shape of the 2nd input is known
  auto circle_rhs = loco::must_cast<luci::CircleNode *>(rhs);
  CHECK_OR_FALSE(circle_rhs->shape_status() == luci::ShapeStatus::VALID);
  // TODO as of 06/23/20 TFLite only supports rank 2 for 2nd input. Fix this once that changes!
  CHECK_OR_FALSE(circle_rhs->rank() == 2);
  // Check that input data type is supported
  CHECK_OR_THROW(lhs_dtype == U8 || lhs_dtype == S16 || lhs_dtype == FLOAT32,
                 "Only UInt8, Int16 and Float32 data types are supported by MatMul");

  if (transpose_a)
  {
    // Create a permutation constant node
    std::vector<int32_t> perm;
    const auto lhs_rank = static_cast<int32_t>(circle_lhs->rank());
    for (int32_t i = 0; i < lhs_rank; ++i)
      perm.push_back(i);
    std::swap(perm[circle_lhs->rank() - 1], perm[circle_lhs->rank() - 2]);
    auto perm_node = luci::create_const_node(graph, S32, {circle_lhs->rank()}, perm);
    perm_node->name(name + "/lhs/Transpose/perm");
    // Now make a transpose node
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(lhs);
    transpose_node->perm(perm_node);
    transpose_node->name(name + "/lhs/Transpose");
    luci::add_origin(transpose_node, luci::get_origin(cop));
    lhs = transpose_node;
  }

  // Transpose the second input if needed. TFLite FullyConnected operator
  // assumes the second input is in column-major order, but the input is
  // in row-major order, thus we need to convert between them.
  if (!transpose_b)
  {
    const std::vector<int32_t> perm{1, 0};
    auto perm_node = luci::create_const_node(graph, S32, {2}, perm);
    perm_node->name(name + "/rhs/Transpose/perm");
    auto transpose_node = graph->nodes()->create<luci::CircleTranspose>();
    transpose_node->a(rhs);
    transpose_node->perm(perm_node);
    transpose_node->name(name + "/rhs/Transpose");
    luci::add_origin(transpose_node, luci::get_origin(cop));
    rhs = transpose_node;
  }

  auto empty_bias = graph->nodes()->create<luci::CircleOutputExclude>();

  auto fc_node = graph->nodes()->create<luci::CircleFullyConnected>();
  fc_node->input(lhs);
  fc_node->weights(rhs);
  fc_node->bias(empty_bias);
  fc_node->fusedActivationFunction(luci::FusedActFunc::NONE);
  fc_node->name(name + "/FullyConnected");
  luci::add_origin(fc_node, luci::get_origin(cop));

  auto customOut = loco::succs(cop);
  assert(customOut.size() == 1);
  replace(*customOut.begin()).with(fc_node);
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
