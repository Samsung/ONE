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
#include "compiler/LoweredGraph.h"
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
      const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
          &lowered_subgs)
      : _lowered_subgs(lowered_subgs), _operands(lowered_subgs.at(subg_idx)->graph().operands()),
        _operations(lowered_subgs.at(subg_idx)->graph().operations()),
        _return_has_dynamic_tensor(false)
  { /* empty */
  }
  virtual ~StaticShapeInferer() = default;

public:
  /**
   * @brief Infer shape of operands beloning to ops and set the output shape.
   *        If output shape cannot be known without running op, mark it so that it can be allocated
   *        when running kernel.
   * @param op_seq sequence of operations
   * @return @c true if op_seq's input or output has any dynamic tensor; @c false otherwise.
   */
  bool infer(const ir::OpSequence &op_seq);

  void dump();

private:
  bool checkDynamicInput(const ir::Operation &op);
  void setDynamicOutput(const ir::Operation &op);

private:
  // TODO Define visitors for operations. List them in alphabetic order.
  void visit(const ir::operation::ArgMax &op) override;
  void visit(const ir::operation::BatchMatMul &op) override;
  void visit(const ir::operation::BCQFullyConnected &op) override;
  void visit(const ir::operation::BCQGather &op) override;
  void visit(const ir::operation::BinaryArithmetic &op) override;
  void visit(const ir::operation::BroadcastTo &op) override;
  void visit(const ir::operation::Comparison &op) override;
  void visit(const ir::operation::Concat &op) override;
  void visit(const ir::operation::Conv2D &op) override;
  void visit(const ir::operation::ElementwiseActivation &op) override;
  void visit(const ir::operation::ElementwiseBinary &op) override;
  void visit(const ir::operation::ElementwiseUnary &op) override;
  void visit(const ir::operation::ExpandDims &op) override;
  void visit(const ir::operation::Fill &op) override;
  void visit(const ir::operation::FullyConnected &op) override;
  void visit(const ir::operation::FusedBatchNorm &op) override;
  void visit(const ir::operation::Gather &op) override;
  void visit(const ir::operation::If &op) override;
  void visit(const ir::operation::L2Normalization &op) override;
  void visit(const ir::operation::MatrixBandPart &op) override;
  void visit(const ir::operation::OneHot &op) override;
  void visit(const ir::operation::Pack &op) override;
  void visit(const ir::operation::Pad &op) override;
  void visit(const ir::operation::Permute &op) override;
  void visit(const ir::operation::Pow &op) override;
  void visit(const ir::operation::Range &op) override;
  void visit(const ir::operation::Reduce &op) override;
  void visit(const ir::operation::Reshape &op) override;
  void visit(const ir::operation::ResizeBilinear &op) override;
  void visit(const ir::operation::Reverse &op) override;
  void visit(const ir::operation::Select &op) override;
  void visit(const ir::operation::Shape &op) override;
  void visit(const ir::operation::Slice &op) override;
  void visit(const ir::operation::Softmax &op) override;
  void visit(const ir::operation::SpaceToBatchND &op) override;
  void visit(const ir::operation::Split &op) override;
  void visit(const ir::operation::Squeeze &op) override;
  void visit(const ir::operation::StridedSlice &op) override;
  void visit(const ir::operation::SquaredDifference &op) override;
  void visit(const ir::operation::Tile &op) override;
  void visit(const ir::operation::Transpose &op) override;
  void visit(const ir::operation::Unpack &op) override;
  void visit(const ir::operation::While &op) override;

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
  const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<compiler::LoweredGraph>>
      &_lowered_subgs;
  // _operands and _operations can be changed by controlflow operation
  ir::Operands &_operands;     // operands of current subgraph
  ir::Operations &_operations; // operations of current subgraph
  bool _return_has_dynamic_tensor;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_STATIC_SHAPE_INFERENCE_H__
