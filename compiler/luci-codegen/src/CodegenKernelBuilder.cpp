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

#include "luci/IR/CircleNodeVisitor.h"
#include "loco/IR/Algorithm.h"

#include <cassert>

namespace luci_codegen
{

namespace
{

struct AddOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b) { return a + b; }
};

struct SubOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b) { return a - b; }
};

struct MulOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b) { return a * b; }
};

struct DivOp
{
  static Halide::Expr op(Halide::Expr a, Halide::Expr b) { return a / b; }
};

std::vector<Halide::Expr> debroadcast_iter_space(const std::vector<Halide::Expr> &output_space,
                                                 const luci::CircleNode *node)
{
  int rank = node->rank();
  std::vector<Halide::Expr> iter_space(rank);
  for (int i = rank - 1; i >= 0; --i)
  {
    if (node->dim(rank - 1 - i) == 1)
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

class CodegenKernelBuilderImpl : public luci::CircleNodeMutableVisitor<void>
{
private:
  SubgraphContext &_subgraph;

  // elementwise operator supports

  template <typename OP> void binary_operator(luci::CircleNode *node);

public:
  explicit CodegenKernelBuilderImpl(SubgraphContext &subgraph);

  void visit(luci::CircleConst *node) override;

  void visit(luci::CircleAdd *node) override;

  void visit(luci::CircleSub *node) override;

  void visit(luci::CircleMul *node) override;

  void visit(luci::CircleDiv *node) override;

  void visit(luci::CircleTanh *node) override;

  void visit(luci::CircleLogistic *node) override;

  void visit(luci::CircleSplit *node) override;

  void visit(luci::CircleSplitOut *node) override;

  void visit(luci::CircleFullyConnected *node) override;

  /// @brief Default fallback
  void visit(luci::CircleNode *) override;
};

CodegenKernelBuilderImpl::CodegenKernelBuilderImpl(SubgraphContext &subgraph) : _subgraph(subgraph) {}

template <typename OP> void CodegenKernelBuilderImpl::binary_operator(luci::CircleNode *node)
{
  std::vector<Halide::Expr> output_vars = iter_space(node);

  luci::CircleNode *arg_a = static_cast<luci::CircleNode *>(node->arg(0));
  std::vector<Halide::Expr> arg_a_vars = debroadcast_iter_space(output_vars, arg_a);
  Halide::Func input_a = _subgraph.get_func(arg_a);

  luci::CircleNode *arg_b = static_cast<luci::CircleNode *>(node->arg(1));
  std::vector<Halide::Expr> arg_b_vars = debroadcast_iter_space(output_vars, arg_b);
  Halide::Func input_b = _subgraph.get_func(arg_b);

  Halide::Func output_func = _subgraph.get_func(node);
  output_func(output_vars) = OP::op(input_a(arg_a_vars), input_b(arg_b_vars));
}

void CodegenKernelBuilderImpl::visit(luci::CircleConst *node)
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
      Halide::Func const_func = _subgraph.get_func(node);
      const_func(iter_space) = buf(iter_space);
    }
  }
}

void CodegenKernelBuilderImpl::visit(luci::CircleAdd *node) { binary_operator<AddOp>(node); }

void CodegenKernelBuilderImpl::visit(luci::CircleSub *node) { binary_operator<SubOp>(node); }

void CodegenKernelBuilderImpl::visit(luci::CircleMul *node) { binary_operator<MulOp>(node); }

void CodegenKernelBuilderImpl::visit(luci::CircleDiv *node) { binary_operator<DivOp>(node); }

void CodegenKernelBuilderImpl::visit(luci::CircleTanh *node)
{
  std::vector<Halide::Expr> iterators = iter_space(node);
  Halide::Func input = _subgraph.get_func(node->x());

  constexpr float min_x = -9.f;
  constexpr float max_x = 9.f;

  constexpr float t_a1 = 4.89352455891786e-03f;
  constexpr float t_a3 = 6.37261928875436e-04;
  constexpr float t_a5 = 1.48572235717979e-05;
  constexpr float t_a7 = 5.12229709037114e-08;
  constexpr float t_a9 = -8.60467152213735e-11;
  constexpr float t_a11 = 2.00018790482477e-13;
  constexpr float t_a13 = -2.76076847742355e-16;

  constexpr float t_b0 = 4.89352518554385e-03;
  constexpr float t_b2 = 2.26843463243900e-03;
  constexpr float t_b4 = 1.18534705686654e-04;
  constexpr float t_b6 = 1.19825839466702e-06;

  Halide::Expr t_clipped_x = Halide::max(Halide::min(input(iterators), max_x), min_x);
  Halide::Expr t_x2 = t_clipped_x * t_clipped_x;

  Halide::Expr t_p1 = t_a13 * t_x2 + t_a11;
  Halide::Expr t_p2 = t_p1 * t_x2 + t_a9;
  Halide::Expr t_p3 = t_p2 * t_x2 + t_a7;
  Halide::Expr t_p4 = t_p3 * t_x2 + t_a5;
  Halide::Expr t_p5 = t_p4 * t_x2 + t_a3;
  Halide::Expr t_p = (t_p5 * t_x2 + t_a1) * t_clipped_x;

  Halide::Expr t_q1 = t_b6 * t_x2 + t_b4;
  Halide::Expr t_q2 = t_q1 * t_x2 + t_b2;
  Halide::Expr t_q = t_q2 * t_x2 + t_b0;

  Halide::Expr tanh = t_p / t_q;
  _subgraph.get_func(node)(iterators) = tanh;
}

void CodegenKernelBuilderImpl::visit(luci::CircleLogistic *node)
{
  std::vector<Halide::Expr> iterators = iter_space(node);
  Halide::Func input = _subgraph.get_func(node->x());

  constexpr float min_x = -18.f;
  constexpr float max_x = 18.f;

  // The monomial coefficients of the numerator polynomial (odd).
  constexpr float s_a1 = 2.48287947061529e-01;
  constexpr float s_a3 = 8.51377133304701e-03;
  constexpr float s_a5 = 6.08574864600143e-05;
  constexpr float s_a7 = 1.15627324459942e-07;
  constexpr float s_a9 = 4.37031012579801e-11;

  // The monomial coefficients of the denominator polynomial (even).
  constexpr float s_b0 = 9.93151921023180e-01;
  constexpr float s_b2 = 1.16817656904453e-01;
  constexpr float s_b4 = 1.70198817374094e-03;
  constexpr float s_b6 = 6.29106785017040e-06;
  constexpr float s_b8 = 5.76102136993427e-09;
  constexpr float s_b10 = 6.10247389755681e-13;

// construct first sigmoid operation
  Halide::Expr s1_clipped_x = Halide::max(Halide::min(input(iterators), max_x), min_x);
  Halide::Expr s1_x2 = s1_clipped_x * s1_clipped_x;

  Halide::Expr s1_p1 = s_a9 * s1_x2 + s_a7;
  Halide::Expr s1_p2 = s1_p1 * s1_x2 + s_a5;
  Halide::Expr s1_p3 = s1_p2 * s1_x2 + s_a3;
  Halide::Expr s1_p = (s1_p3 * s1_x2 + s_a1) * s1_clipped_x;

  Halide::Expr s1_q1 = s_b10 * s1_x2 + s_b8;
  Halide::Expr s1_q2 = s1_q1 * s1_x2 + s_b6;
  Halide::Expr s1_q3 = s1_q2 * s1_x2 + s_b4;
  Halide::Expr s1_q4 = s1_q3 * s1_x2 + s_b2;
  Halide::Expr s1_q = s1_q4 * s1_x2 + s_b0;

  Halide::Expr sigmoid = s1_p / s1_q + 0.5f;
  _subgraph.get_func(node)(iterators) = sigmoid;
}

void CodegenKernelBuilderImpl::visit(luci::CircleSplit *node)
{
  // nothing to do. everything will be done on OUT nodes
}

void CodegenKernelBuilderImpl::visit(luci::CircleSplitOut *node)
{
  auto split_node = static_cast<luci::CircleSplit *>(node->input());
  auto split_input_node = static_cast<luci::CircleNode *>(split_node->input());
  assert(static_cast<luci::CircleNode *>(split_node->split_dim())->opcode() == luci::CircleOpcode::CIRCLECONST);
  auto split_dim_node = static_cast<luci::CircleConst *>(split_node->split_dim());

  assert(split_dim_node->dtype() == loco::DataType::S32);
  assert(split_dim_node->size<loco::DataType::S32>() == 1);

  int split_dim = split_dim_node->at<loco::DataType::S32>(0);
  int split_input_dim_size = split_node->dim(split_dim).value(); // dim size before split
  int split_output_dim_size = split_input_dim_size / split_node->num_split(); // dim size after split

  assert(split_input_dim_size % split_node->num_split() == 0);

  int start_tile_index = split_output_dim_size * node->index();

  auto output_iterators = iter_space(node);

  auto input_iterators = output_iterators;
  input_iterators[split_node->rank() - 1 - split_dim] += start_tile_index;

  Halide::Func input = _subgraph.get_func(split_input_node);

  Halide::Func split_func = _subgraph.get_func(node);

  split_func(output_iterators) = input(input_iterators);
}

void CodegenKernelBuilderImpl::visit(luci::CircleFullyConnected *node)
{
  assert(node->weights_format() == luci::CircleFullyConnected::WeightsFormat::DEFAULT);
  assert(node->rank() == 2);
  assert(node->dim(0) == 1);
  Halide::Func input = _subgraph.get_func(node->input());
  Halide::Func fc = _subgraph.get_func(node);
  Halide::Func weights = _subgraph.get_func(node->weights());
  Halide::Func bias = _subgraph.get_func(node->bias());
  Halide::Var output_iter;
  Halide::RDom partial_sum_iter(0, static_cast<int>(node->dim(1).value()), "partial_sum_iter");


  fc(output_iter, 0) = bias(output_iter);
  fc(output_iter, 0) += weights(partial_sum_iter, output_iter) * input(partial_sum_iter, 0);
}

void CodegenKernelBuilderImpl::visit(luci::CircleNode *)
{
  INTERNAL_EXN("CodegenKernelBuilder: unsupported node");
}

} // unnamed namespace

CodegenKernelBuilder::CodegenKernelBuilder(SubgraphContext &subgraph) : _subgraph(subgraph) {}

void CodegenKernelBuilder::process()
{
  std::vector<luci::CircleNode *> sorted_nodes;
  // collect subgraph outputs
  std::vector<loco::Node *> outputs;
  for (auto node: _subgraph.get_outputs())
  {
    outputs.push_back(node.first);
  }
  // collect nodes in topological order
  for (auto node: loco::postorder_traversal(outputs))
  {
    luci::CircleNode *circle_node = static_cast<luci::CircleNode *>(node);
    if (_subgraph.contains(circle_node))
    {
      sorted_nodes.push_back(circle_node);
    }
  }

  // Define kernels
  CodegenKernelBuilderImpl visitor(_subgraph);
  for (auto node: sorted_nodes)
  {
    node->accept(&visitor);
  }
}

static bool is_supported_fc(luci::CircleFullyConnected *fc)
{
  int outputs = fc->dim(0).value();
  return outputs == 1 && fc->shape_status() == luci::ShapeStatus::VALID &&
      fc->weights_format() == luci::CircleFullyConnected::WeightsFormat::DEFAULT;
}

static bool is_supported_split(luci::CircleSplit *split)
{
  bool const_split_dim = static_cast<luci::CircleNode *>(split->split_dim())->opcode() == luci::CircleOpcode::CIRCLECONST;
  if (!const_split_dim)
    return false;
  auto split_dim = static_cast<luci::CircleConst *>(split->split_dim());
  bool supported_split_dim_dtype = (split_dim->dtype() == loco::DataType::S32);
  if (!supported_split_dim_dtype)
    return false;
  bool is_scalar_dim = (split_dim->size<loco::DataType::S32>() == 1);
  if (!is_scalar_dim)
    return false;

  int split_dim_value = split_dim->at<loco::DataType::S32>(0);
  int split_input_dim_size = split->dim(split_dim_value).value(); // dim size before split

  if (split_input_dim_size % split->num_split() != 0)
    return false;

  return split->shape_status() == luci::ShapeStatus::VALID;
}

bool CodegenKernelBuilder::is_supported(luci::CircleNode *node)
{
  assert(dynamic_cast<luci::CircleNode *>(node));
  bool is_quantized = node->quantparam();
  if (is_quantized)
    return false;
  luci::CircleNode *circle_node = static_cast<luci::CircleNode *>(node);
  switch (circle_node->opcode())
  {
    case luci::CircleOpcode::CIRCLECONST:
    case luci::CircleOpcode::ADD:
    case luci::CircleOpcode::SUB:
    case luci::CircleOpcode::MUL:
    case luci::CircleOpcode::DIV:
      return circle_node->shape_status() == luci::ShapeStatus::VALID;
    case luci::CircleOpcode::TANH:
    case luci::CircleOpcode::LOGISTIC:
      return circle_node->dtype() == loco::DataType::FLOAT32 &&
             circle_node->shape_status() == luci::ShapeStatus::VALID;
    case luci::CircleOpcode::FULLY_CONNECTED:
      return is_supported_fc(static_cast<luci::CircleFullyConnected *>(node));
    case luci::CircleOpcode::SPLIT:
      return is_supported_split(static_cast<luci::CircleSplit *>(node));
    case luci::CircleOpcode::CIRCLESPLITOUT:
      return is_supported_split(static_cast<luci::CircleSplit *>(node->arg(0)));
  }
  return false;
}

} // luci_codegen
