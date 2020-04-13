/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "moco/Pass/Passes/ConstantFoldPack.h"

#include "ConstantFoldHelper.h"
#include "TensorPackEnumerator.h"

#include <moco/IR/Nodes/TFPack.h>
#include <moco/IR/Nodes/TFConst.h>

#include <moco/Support/NodeAs.h>

#include <oops/UserExn.h>

#include <cassert>
#include <vector>

namespace
{

// TODO move to loco
bool operator==(const loco::TensorShape &lhs, const loco::TensorShape &rhs)
{
  if (lhs.rank() != rhs.rank())
    return false;
  for (uint32_t axis = 0; axis < lhs.rank(); ++axis)
  {
    if (!(lhs.dim(axis) == rhs.dim(axis)))
      return false;
  }
  return true;
}

bool valid_axis_range(int32_t output_rank, int32_t pack_axis)
{
  // check axis range in [-r-1, r+1)
  assert(output_rank > 0);
  return (-output_rank <= pack_axis) && (pack_axis < output_rank);
}

bool constantfold_pack(moco::TFPack *node)
{
  // check if all the inputs are Const
  std::vector<moco::TFConst *> input_nodes;
  uint32_t num = node->N();

  for (uint32_t index = 0; index < num; ++index)
  {
    auto in = dynamic_cast<moco::TFConst *>(node->values(index));
    if (in == nullptr)
      return false;

    input_nodes.push_back(in);
  }
  assert(input_nodes.size() == num);

  // check if all inputs have same shape and dtype
  auto input_0 = input_nodes.at(0);
  auto shape_0 = moco::tensor_shape(input_0);
  auto dtype_0 = input_0->dtype();
  if (dtype_0 != loco::DataType::S32 && dtype_0 != loco::DataType::FLOAT32)
  {
    // TODO support other types
    assert(false);
    return false;
  }
  for (uint32_t index = 1; index < num; ++index)
  {
    auto input_i = input_nodes.at(index);
    auto shape_i = moco::tensor_shape(input_i);
    auto dtype_i = input_i->dtype();
    if (!(shape_0 == shape_i))
      return false;
    if (dtype_0 != dtype_i)
      return false;
  }

  int32_t output_rank = static_cast<int32_t>(shape_0.rank() + 1);
  int32_t pack_axis = node->axis();
  if (!valid_axis_range(output_rank, pack_axis))
  {
    throw oops::UserExn("axis is out of range: ", node->name());
  }

  if (pack_axis < 0)
  {
    pack_axis = output_rank + pack_axis;
  }

  // define output shape
  loco::TensorShape output_shape;
  output_shape.rank(output_rank);

  for (int32_t r = 0, s = 0; r < output_rank; ++r)
  {
    if (r == pack_axis)
    {
      output_shape.dim(r).set(num);
    }
    else
    {
      output_shape.dim(r).set(shape_0.dim(s++).value());
    }
  }

  auto graph = node->graph();

  // create new constant
  auto output_const = moco::new_const(graph, output_shape, input_0->dtype());

  moco::TensorPackEnumerator etor;

  etor.shape(shape_0, output_shape);
  etor.axis(pack_axis);
  for (etor.start(); etor.valid(); etor.advance())
  {
    uint32_t inp_num = etor.inp_num();
    uint32_t inp_element = etor.inp_element();
    uint32_t out_element = etor.out_element();

    auto inp_const = input_nodes[inp_num];

    if (input_0->dtype() == loco::DataType::S32)
    {
      int32_t val = inp_const->at<loco::DataType::S32>(inp_element);
      output_const->at<loco::DataType::S32>(out_element) = val;
    }
    else if (input_0->dtype() == loco::DataType::FLOAT32)
    {
      float val = inp_const->at<loco::DataType::FLOAT32>(inp_element);
      output_const->at<loco::DataType::FLOAT32>(out_element) = val;
    }
  }

  // replace
  loco::replace(node).with(output_const);

  return true;
}

} // namespace

namespace moco
{

/**
 * @note This will Replace TFPack with TFConst when inputs are TFConst
 *
 *       Before
 *                 A --- TFPack --- C
 *                 B --/
 *       After
 *                 A --- TFPack
 *                 B --/
 *                       TFConst ---------- C
 *       Where
 *                 A, B : inputs of TFPack
 *                 C : a node that uses TFPack as an input
 *                 TFPack is disconnected from C
 *                 Nodes are drawn multiple times to simplify the diagram
 */
bool ConstantFoldPack::run(loco::Graph *graph)
{
  bool changed = false;
  for (auto node : loco::active_nodes(loco::output_nodes(graph)))
  {
    if (auto pack_node = as<moco::TFPack>(node))
    {
      if (constantfold_pack(pack_node))
        changed = true;
    }
  }

  return changed;
}

} // namespace moco
