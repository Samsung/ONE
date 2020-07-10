/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ONERT_COMPILER_STATIC_SHAPE_INFERENCE_H__
#define __ONERT_COMPILER_STATIC_SHAPE_INFERENCE_H__

#include "ir/OperationVisitor.h"
#include "ir/OpSequence.h"
#include "ir/LoweredGraph.h"
#include "ir/Index.h"

#include <memory>
#include <unordered_map>

namespace onert
{
namespace compiler
{

/**
 * @brief Class to infer shape before running kernels. It does the following:
 *        - re-calculate and set output shape at compile time (before running kernels)
 *        - if calculation cannot be done at compile time, mark the outputs to be dynamic, meaning
 *          shapes of outputs will be calculated during running kernels
 */
class StaticShapeInferer : public ir::OperationVisitor
{
public:
  StaticShapeInferer(
      const ir::SubgraphIndex &subg_idx,
      const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<ir::LoweredGraph>> &lowered_subgs)
      : _lowered_subgs(lowered_subgs), _operands(lowered_subgs.at(subg_idx)->graph().operands()),
        _operations(lowered_subgs.at(subg_idx)->graph().operations())
  { /* empty */
  }
  virtual ~StaticShapeInferer() = default;

public:
  /**
   * @brief Infer shape of operands beloning to ops and set the output shape.
   *        If output shape cannot be known without running op, mark it so that it can be allocated
   *        when running kernel.
   * @param op_seq sequence of operations
   */
  void infer(const ir::OpSequence &op_seq)
  {
    for (const auto &operation_idx : op_seq.operations())
    {
      _operations.at(operation_idx).accept(*this);
    }
  }

  void dump();

private:
  // TODO Define visitors for operations. List them in alphabetic order.
  void visit(const ir::operation::Abs &op);
  void visit(const ir::operation::Add &op);
  void visit(const ir::operation::ArgMax &op);
  void visit(const ir::operation::BatchMatMul &op);
  void visit(const ir::operation::BroadcastTo &op);
  void visit(const ir::operation::Cast &op);
  void visit(const ir::operation::Comparison &op);
  void visit(const ir::operation::Concat &op);
  void visit(const ir::operation::Conv2D &op);
  void visit(const ir::operation::Cos &op);
  void visit(const ir::operation::Div &op);
  void visit(const ir::operation::Exp &op);
  void visit(const ir::operation::ExpandDims &op);
  void visit(const ir::operation::Fill &op);
  void visit(const ir::operation::FullyConnected &op);
  void visit(const ir::operation::FusedBatchNorm &op);
  void visit(const ir::operation::Gather &op);
  void visit(const ir::operation::If &op);
  void visit(const ir::operation::Log &op);
  void visit(const ir::operation::LogicalNot &op);
  void visit(const ir::operation::LogicalOr &op);
  void visit(const ir::operation::Logistic &op);
  void visit(const ir::operation::MatrixBandPart &op);
  void visit(const ir::operation::Max &op);
  void visit(const ir::operation::Mean &op);
  void visit(const ir::operation::Min &op);
  void visit(const ir::operation::Mul &op);
  void visit(const ir::operation::Neg &op);
  void visit(const ir::operation::OneHot &op);
  void visit(const ir::operation::Pack &op);
  void visit(const ir::operation::Pad &op);
  void visit(const ir::operation::Permute &op);
  void visit(const ir::operation::Pow &op);
  void visit(const ir::operation::Range &op);
  void visit(const ir::operation::ReduceAll &op);
  void visit(const ir::operation::ReduceMin &op);
  void visit(const ir::operation::ReduceProd &op);
  void visit(const ir::operation::ReduceSum &op);
  void visit(const ir::operation::Reshape &op);
  void visit(const ir::operation::Round &op);
  void visit(const ir::operation::RSQRT &op);
  void visit(const ir::operation::Reverse &op);
  void visit(const ir::operation::Select &op);
  void visit(const ir::operation::Shape &op);
  void visit(const ir::operation::Sin &op);
  void visit(const ir::operation::Slice &op);
  void visit(const ir::operation::Softmax &op);
  void visit(const ir::operation::Split &op);
  void visit(const ir::operation::Squeeze &op);
  void visit(const ir::operation::StridedSlice &op);
  void visit(const ir::operation::Sub &op);
  void visit(const ir::operation::SquaredDifference &op);
  void visit(const ir::operation::Tanh &op);
  void visit(const ir::operation::Tile &op);
  void visit(const ir::operation::Transpose &op);
  void visit(const ir::operation::Unpack &op);
  void visit(const ir::operation::While &op);
  void visit(const ir::operation::ZerosLike &op);

private:
  /**
   * @brief Performs shape inference for arithmetic operation
   */
  void handleBinaryArithmeticOp(const ir::Operation &op, const ir::OperandIndex lhs_idx,
                                const ir::OperandIndex rhs_idx);

  /**
   * @brief Performs shape inference for unary op whose output shape is
   *        always same with input shape
   */
  void handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_idx);

private:
  const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<ir::LoweredGraph>> &_lowered_subgs;
  // _operands and _operations can be changed by controlflow operation
  ir::Operands &_operands;     // operands of current subgraph
  ir::Operations &_operations; // operations of current subgraph
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_STATIC_SHAPE_INFERENCE_H__
