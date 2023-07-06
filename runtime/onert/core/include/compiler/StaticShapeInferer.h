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

#ifndef __ONERT_COMPILER_STATIC_SHAPE_INFERER_H__
#define __ONERT_COMPILER_STATIC_SHAPE_INFERER_H__

#include "ir/OperationVisitor.h"
#include "compiler/LoweredGraph.h"
#include "ir/Index.h"

#include <memory>
#include <unordered_map>

namespace onert
{
namespace compiler
{
/**
 * @brief Class that observe and update operands.
 */
class OperandObserver
{
public:
  /**
   * @brief Constructor of OperandObserver
   *
   * @param operands Operands to be updated
   */
  OperandObserver(const std::vector<ir::Operand *> &operands) : _operands{operands} {}
  /**
   * @brief Destructor of OperandObserver
   */
  virtual ~OperandObserver() = default;

public:
  /**
   * @brief Update Shape and some OperandInfo of operands
   *
   * @param operands Operands to be updated
   * @param unpredictable Whether runtime can predict shapes of operands in compilation time
   */
  void updateShapes(const std::vector<ir::OperandInfo> &changed_operands_info,
                    bool unpredictable = false);

private:
  std::vector<ir::Operand *> _operands;
};

/**
 * @brief Class to infer shape before running kernels. It does the following:
 *        - re-calculate and set output shape at compile time (before running kernels)
 *        - if calculation cannot be done at compile time, mark the outputs to be dynamic, meaning
 *          shapes of outputs will be calculated during running kernels
 */
class StaticShapeInferer : public ir::OperationVisitor
{
public:
  StaticShapeInferer(compiler::ILoweredGraph *lowered_subg)
    : _lowered_subg{lowered_subg}, _subg_input_observers{}, _controlflow_output_observer{nullptr},
      _child_inferers{}
  {
  }
  virtual ~StaticShapeInferer() = default;

public:
  void appendSubgInputObserver(const ir::SubgraphIndex &subg_idx,
                               std::unique_ptr<OperandObserver> &&subg_input_observer) noexcept
  {
    _subg_input_observers[subg_idx] = std::move(subg_input_observer);
  }

  void setControlflowOutputObserver(std::unique_ptr<OperandObserver> &&output_observer) noexcept
  {
    _controlflow_output_observer = std::move(output_observer);
  }

  void appendChildInferer(const ir::SubgraphIndex &subg_idx, compiler::StaticShapeInferer *inferer)
  {
    _child_inferers[subg_idx] = inferer;
  }

  /**
   * @brief Infer shape of operands belonging to ops and set the output shape.
   *        If output shape cannot be known without running op, mark it so that it can be allocated
   *        when running kernel.
   */
  void infer(void);

  void dump();

  /**
   * @brief     Create a shape inferer map for a lowered model
   * @param[in] lowered_subgs lowered model map
   * @return    Shape inferer map
   */
  static std::unordered_map<ir::SubgraphIndex, std::unique_ptr<StaticShapeInferer>>
  createStaticShapeInferers(
    const std::unordered_map<ir::SubgraphIndex, ILoweredGraph *> &lowered_subgs);

private:
  bool checkDynamicInput(const ir::IOperation &op);
  bool checkDynamicOutput(const ir::IOperation &op);
  void setDynamicOutput(const ir::IOperation &op);

private:
  // TODO Define visitors for operations. List them in alphabetic order.
  void visit(const ir::operation::ArgMinMax &op) override;
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
  void visit(const ir::operation::LSTM &op) override;
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
  void visit(const ir::operation::DetectionPostProcess &op) override;
  void visit(const ir::operation::Bulk &op) override;

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
  compiler::ILoweredGraph *_lowered_subg;
  std::unordered_map<ir::SubgraphIndex, std::unique_ptr<OperandObserver>>
    _subg_input_observers;                                       // child subg input
  std::unique_ptr<OperandObserver> _controlflow_output_observer; // parent controlflow op output
  std::unordered_map<ir::SubgraphIndex, compiler::StaticShapeInferer *> _child_inferers;
};

} // namespace compiler
} // namespace onert

#endif // __ONERT_COMPILER_STATIC_SHAPE_INFERER_H__
