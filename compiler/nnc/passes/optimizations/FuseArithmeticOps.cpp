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

#include "passes/optimizations/FuseArithmeticOps.h"
#include "passes/optimizations/OptimizationUtils.h"
#include "mir/ops/AddOp.h"
#include "mir/ops/ConstantOp.h"
#include "mir/ops/Conv2DOp.h"
#include "mir/ops/MulOp.h"
#include "mir/Graph.h"
#include "mir/Tensor.h"
#include "mir/Index.h"
#include "mir/TensorVariant.h"
#include "mir/ShapeRange.h"

#include <algorithm>

namespace nnc
{

namespace
{

using namespace mir;
using namespace std;
using namespace opt_util;

using OpType = Operation::Type;
using Edge = pair<Operation *, Operation *>;

/**
 * This function used to get 'ConstantOp' with weights of 'AddOp', 'MulOp' or 'Conv2DOp'
 * For each of these ops weights stored in second input node
 */
ops::ConstantOp *getSecondInputAsConst(Operation *op)
{
  assert(op->getType() == OpType::add || op->getType() == OpType::mul ||
         op->getType() == OpType::conv2D);
  return dynamic_cast<ops::ConstantOp *>(op->getInput(1)->getNode());
}

// This function finds successive operations of given types, with ConstantOp as second input
vector<Edge> findSuccessiveOpsWithConstWeights(Graph *g, OpType first_op_type,
                                               OpType second_op_type)
{
  vector<Edge> matches;
  unordered_set<Operation *> matched_nodes;
  for (auto *first_op : g->getNodes())
  {
    if (first_op->getType() == first_op_type && getSecondInputAsConst(first_op))
    {
      for (auto &out : first_op->getOutputs())
      {
        for (Operation::Use use : out.getUses())
        {
          Operation *second_op = use.getNode();
          if (second_op->getType() == second_op_type && getSecondInputAsConst(second_op))
          {
            /**
             * Don't match already matched nodes, so for op1->op2->op3 this function
             * will return {{f1, f2}} and not {{f1, f2}, {f2, f3}}
             */
            if (matched_nodes.find(first_op) == matched_nodes.end() &&
                matched_nodes.find(second_op) == matched_nodes.end())
            {
              matched_nodes.emplace(first_op);
              matched_nodes.emplace(second_op);
              matches.emplace_back(first_op, second_op);
            }
          }
        }
      }
    }
  }
  return matches;
}

/**
 * This function merges two ConstantOp into new one, by elementwise multiplication or addition
 * If first ConstantOp rank > 1, second one broadcasting to first by axis=0
 */
Operation *mergeConstantOps(Graph *g, const ops::ConstantOp *const1_op,
                            const ops::ConstantOp *const2_op, OpType merge_type)
{
  const auto &const1_val = const1_op->getValue();
  const auto &const2_val = const2_op->getValue();
  assert(const1_val.getShape().rank() >= const2_val.getShape().rank());
  assert(const2_val.getShape().rank() == 1);
  assert(const1_val.getShape().dim(0) == const2_val.getShape().dim(0));

  // Create and fill TensorVariant for new ConstantOp
  TensorVariant new_const_val(DataType::FLOAT32, const1_val.getShape());
  Tensor<float> const1_accessor(const1_val);
  Tensor<float> const2_accessor(const2_val);
  Tensor<float> new_const_accessor(new_const_val);
  ShapeRange const1_range(const1_val.getShape());
  for (auto &idx : const1_range)
  {
    float operand1 = const1_accessor.at(idx);
    /**
     * Broadcast second ConstantOp to first one:
     * idx of second constant always has rank 1 and equals to first dimension of first constant idx
     */
    float operand2 = const2_accessor.at(Index{idx.at(0)});
    switch (merge_type)
    {
      case OpType::mul:
        new_const_accessor.at(idx) = operand1 * operand2;
        break;
      case OpType::add:
        new_const_accessor.at(idx) = operand1 + operand2;
        break;
      default:
        assert(false && "only 'mul' and 'add' constants merge types supported");
    }
  }

  return g->create<ops::ConstantOp>(new_const_val);
}

// TODO: support 'DepthwiseConv'->'Mul'
/**
 * This function fuses some successive operations with constant weights into one:
 * 'Add'->'Add' into 'Add'; 'Mul'->'Mul' into 'Mul'; 'Conv'->'Mul' into 'Conv';
 * Before:                  | After:
 * -------------------------|---------------------------
 *  [input] [Const1]        | [input] [Const1*Const2]
 *       \\ //              |      \\ //
 *       [Mul] [Const2]     |      [Mul]
 *          \\ //           |
 *          [Mul]           |
 * -------------------------|---------------------------
 *  [input] [Const1]        | [input] [Const1+Const2]
 *       \\ //              |      \\ //
 *       [Add] [Const2]     |      [Add]
 *          \\ //           |
 *          [Add]           |
 * -------------------------|---------------------------
 *  [input]     [Const1]    | [input]    [Const1*Const2]
 *       \\     //          |      \\    //
 *       [Conv2D] [Const2]  |      [Conv2D]
 *             \\ //        |
 *             [Mul]        |
 */
bool fuseSuccessiveOps(Graph *g)
{
  // Find all successive ops
  vector<Edge> successive_ops;
  auto mul_mul_vec = findSuccessiveOpsWithConstWeights(g, OpType::mul, OpType::mul);
  successive_ops.insert(successive_ops.end(), mul_mul_vec.begin(), mul_mul_vec.end());
  auto add_add_vec = findSuccessiveOpsWithConstWeights(g, OpType::add, OpType::add);
  successive_ops.insert(successive_ops.end(), add_add_vec.begin(), add_add_vec.end());
  auto conv_mul_vec = findSuccessiveOpsWithConstWeights(g, OpType::conv2D, OpType::mul);
  successive_ops.insert(successive_ops.end(), conv_mul_vec.begin(), conv_mul_vec.end());

  for (auto &edge : successive_ops)
  {
    auto const1_op = getSecondInputAsConst(edge.first);
    auto const2_op = getSecondInputAsConst(edge.second);
    assert(const1_op && const2_op);

    // Create new constant operation and copy first successive operation
    auto new_const_op = mergeConstantOps(g, const1_op, const2_op, edge.second->getType());
    auto first_op_input = edge.first->getInput(0);
    auto new_op = g->copyOpWithInputs(edge.first, {first_op_input, new_const_op->getOutput(0)});

    // Replace second successive operation with new one and remove old nodes
    g->replaceNode(edge.second, new_op);
    removeNodeIfUnused(g, edge.first);
    removeNodeIfUnused(g, const1_op);
    removeNodeIfUnused(g, const2_op);
  }

  // If there is no successive operations to fuse - graph wasn't changed
  return !successive_ops.empty();
}

/**
 * This function sinks 'Add' through 'Mul'
 * by multiplying 'Add' weights on 'Mul' weights
 * Before:                  | After:
 *--------------------------|--------------------------
 * [input] [Const1]         | [input] [Const2]
 *      \\ //               |      \\ //
 *      [Add] [Const2]      |      [Mul] [Const1*Const2]
 *         \\ //            |         \\ //
 *         [Mul]            |         [Add]
 *                          |
 */
bool sinkAddThroughMul(Graph *g)
{
  auto add_mul_edges = findSuccessiveOpsWithConstWeights(g, OpType::add, OpType::mul);

  for (auto &edge : add_mul_edges)
  {
    auto old_add_op = edge.first;
    auto old_mul_op = edge.second;
    auto old_add_const_op = getSecondInputAsConst(old_add_op);
    auto ols_mul_const_op = getSecondInputAsConst(old_mul_op);
    assert(old_add_const_op && ols_mul_const_op);

    // Create new operations
    auto old_add_input = old_add_op->getInput(0);
    auto new_mul_op =
      g->copyOpWithInputs(old_mul_op, {old_add_input, ols_mul_const_op->getOutput(0)});
    auto new_add_const_op = mergeConstantOps(g, old_add_const_op, ols_mul_const_op, OpType::mul);
    auto new_add_op =
      g->copyOpWithInputs(old_add_op, {new_mul_op->getOutput(0), new_add_const_op->getOutput(0)});

    // Replace old mul with new add and remove old nodes
    g->replaceNode(old_mul_op, new_add_op);
    removeNodeIfUnused(g, old_add_op);
    removeNodeIfUnused(g, old_add_const_op);
  }

  // If there is no add-mul edges - graph wasn't changed
  return !add_mul_edges.empty();
}

} // unnamed namespace

nnc::PassData nnc::FuseArithmeticOps::run(nnc::PassData data)
{
  auto g = static_cast<Graph *>(data);

  bool graph_changed = true;
  while (graph_changed)
  {
    graph_changed = false;
    graph_changed |= fuseSuccessiveOps(g);
    graph_changed |= sinkAddThroughMul(g);
  }

  return g;
}

} // namespace nnc
