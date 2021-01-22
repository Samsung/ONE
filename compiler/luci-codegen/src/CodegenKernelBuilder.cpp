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

#include "CodegenKernelBuilder.h"
#include "SubgraphContext.h"
#include "Utilities.h"

#include <cassert>

namespace luci_codegen
{

namespace
{

struct AddOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b)
  {
    return a + b;
  }
};

struct SubOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b)
  {
    return a - b;
  }
};

struct MulOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b)
  {
    return a * b;
  }
};

struct DivOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b)
  {
    return a / b;
  }
};

Halide::Type transform_type(loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      return Halide::Type(Halide::Type::Float, 32, 1);
    case loco::DataType::FLOAT64:
      return Halide::Type(Halide::Type::Float, 64, 1);
    case loco::DataType::S32:
      return Halide::Type(Halide::Type::Int, 32, 1);
    case loco::DataType::S64:
      return Halide::Type(Halide::Type::Int, 64, 1);
    default:
      assert("NYI");
  }
  return Halide::Type();
}

std::vector<Halide::Expr> debroadcast_iter_space(const std::vector<Halide::Expr> &output_space, const luci::CircleNode *node)
{
  int rank = node->rank();
  std::vector<Halide::Expr> iter_space(rank);
  for (int i = rank-1; i >= 0; --i)
  {
    if (node->dim(i) == 1)
      iter_space[i] = Halide::Expr(0);
    else
      iter_space[i] = output_space[i];
  }
  return iter_space;
}

std::vector<Halide::Expr> iter_space(const luci::CircleNode *node)
{
  int rank = node->rank();
  std::vector<Halide::Expr> iter_space(rank);
  for (int i = 0; i < rank; ++i)
    iter_space[i] = Halide::Var();
  return iter_space;
}

} // unnamed namespace

CodegenKernelBuilder::CodegenKernelBuilder(SubgraphContext &subgraph) : _subgraph(subgraph) {}

template <typename OP>
void CodegenKernelBuilder::binary_operator(luci::CircleNode *node)
{
  auto rank = node->rank();
  std::vector<Halide::Expr> output_vars = iter_space(node);
  std::vector<Halide::Expr> arg_a_vars = debroadcast_iter_space(output_vars, node); // separate variables according broadcasting rules
  std::vector<Halide::Expr> arg_b_vars = debroadcast_iter_space(output_vars, node);
  Halide::Func output_func;
  Halide::Func input_a = get_func(static_cast<luci::CircleNode *>(node->arg(0)));
  Halide::Func input_b = get_func(static_cast<luci::CircleNode *>(node->arg(1)));

  output_func(output_vars) = OP::op(input_a(arg_a_vars), input_b(arg_b_vars));
  _subgraph.generated_funcs()[node] = output_func;
}

Halide::Func CodegenKernelBuilder::get_func(luci::CircleNode *node)
{
  if (_subgraph.generated_funcs().count(node))
    return _subgraph.generated_funcs()[node];
  // No function found, need to create input
  Halide::ImageParam input = Halide::ImageParam(transform_type(node->dtype()), node->rank());
  _subgraph.add_input(input);
  return input;
}

void CodegenKernelBuilder::visit(luci::CircleConst *node)
{
  size_t rank = node->rank();
  std::vector<int> dims(rank);
  for (int i = 0; i < rank; ++i)
    dims[i] = node->dim(rank - i - 1).value();
  switch (node->dtype())
  {
    case loco::DataType::FLOAT32:
    {
      Halide::Buffer<float> buf(dims);
      size_t size = node->size<loco::DataType::FLOAT32>();
      std::vector<Halide::Expr> iter_space(rank);
      for (int i = 0; i < size; ++i)
      {
        buf.data()[i] = node->at<loco::DataType::FLOAT32>(i);
        iter_space[i] = Halide::Var();
      }
      _subgraph.generated_funcs()[node] = Halide::Func(buf(iter_space));
    }
  }
}

void CodegenKernelBuilder::visit(luci::CircleAdd *node)
{
  binary_operator<AddOp>(node);
}

void CodegenKernelBuilder::visit(luci::CircleSub *node)
{
  binary_operator<SubOp>(node);
}

void CodegenKernelBuilder::visit(luci::CircleMul *node)
{
  binary_operator<MulOp>(node);
}

void CodegenKernelBuilder::visit(luci::CircleDiv *node)
{
  binary_operator<DivOp>(node);
}

void CodegenKernelBuilder::visit(luci::CircleNode *)
{
  INTERNAL_EXN("CodegenKernelBuilder: unsupported node");
}

bool is_supported(luci::CircleNode *node)
{
  assert(dynamic_cast<luci::CircleNode *>(node));
  luci::CircleNode *circle_node = static_cast<luci::CircleNode *>(node);
  switch (circle_node->opcode())
  {
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::ADD:
    case luci::CircleOpcode::SUB:
    case luci::CircleOpcode::MUL:
    case luci::CircleOpcode::DIV:
      return circle_node->shape_status() == luci::ShapeStatus::VALID;
  }
  return false;
}

} // luci_codegen
