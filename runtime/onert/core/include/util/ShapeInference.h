/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef __ONERT_GRAPH_SHAPE_INFERENCE_H__
#define __ONERT_GRAPH_SHAPE_INFERENCE_H__

#include "Utils.h"

#include "ir/operation/AvgPool2D.h"
#include "ir/operation/Concat.h"
#include "ir/operation/MaxPool2D.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/Reshape.h"
#include "ir/operation/RSQRT.h"
#include "ir/operation/StridedSlice.h"
#include "ir/LoweredGraph.h"
#include "ir/Index.h"
#include "ir/Layout.h"
#include "ir/OperationVisitor.h"
#include "backend/IDynamicTensorManager.h"
#include "backend/ITensor.h"
#include "backend/ITensorRegistry.h"

namespace onert
{
namespace shape_inference
{

using Shapes = std::vector<ir::Shape>;

// Define shape calculation for operations. List them in alphabetic order.
// Remove TODO when the function name matching the alphabet is added

ir::Shape inferAvgPoolShape(const ir::Shape &in_shape, const ir::operation::AvgPool2D::Param &param,
                            ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param);

ir::Shape inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                           const ir::operation::Conv2D::Param &param,
                           ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                                    const ir::operation::DepthwiseConv2D::Param &param,
                                    ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape);

ir::Shape inferExpandDimsShape(const ir::Shape &in_shape, int32_t axis);

ir::Shape inferFillShape(const ir::Shape &in_shape, const int32_t *buf);

ir::Shape inferFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &ker_shape);

// TODO write op starting from G
// TODO write op starting from L

ir::Shape inferMaxPoolShape(const ir::Shape &in_shape, const ir::operation::MaxPool2D::Param &param,
                            ir::Layout layout = ir::Layout::NHWC);

// TODO write op starting from N
// TODO write op starting from P
ir::Shape inferReduceShapes(const ir::Shape &input_shape, const std::vector<int> &axes,
                            bool keep_dims);

template <float *> ir::Shape inferRangeShape(float *start_val, float *limit_val, float *delta_val);

template <typename T> ir::Shape inferRangeShape(T start_val, T limit_val, T delta_val);

// TODO write op starting from S
ir::Shape inferTransposeShape(const ir::Shape &in_shape, const std::vector<int> &perm);
// TODO write op starting from U
// TODO write op starting from Z

std::pair<int, int> calcConvLikeHeightAndWidth(const int in_h, const int in_w, const int ker_h,
                                               const int ker_w, const ir::Padding pad,
                                               const ir::Stride stride);

/**
 * @brief Class to infer shape before running kernels. It does the following:
 *        - re-calculate and set output shape at compile time (before running kernels)
 *        - if calculation cannot be done at compile time, mark the outputs to be dynamic, meaning
 *          shapes of outputs will be calculated during running kernels
 */
class StaticInferer : public ir::OperationVisitor
{
public:
  StaticInferer(
      const ir::SubgraphIndex &subg_idx,
      const std::unordered_map<ir::SubgraphIndex, std::unique_ptr<ir::LoweredGraph>> &lowered_subgs)
      : _lowered_subgs(lowered_subgs), _operands(lowered_subgs.at(subg_idx)->graph().operands()),
        _operations(lowered_subgs.at(subg_idx)->graph().operations())
  { /* empty */
  }
  virtual ~StaticInferer() = default;

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
  // Remove TODO when any op starting from the alphabet is added
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
  // TODO write op starting from Q
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
  // TODO write op starting from V
  void visit(const ir::operation::While &op);
  // TODO write op starting from Z
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

// TODO After implement several Ops, check if this class can be merged with StaticInferer
/**
 * @brief Class to infer shape of output tensor at execution time and
 *        allocate memory fo output tensor if needed
 */
class DynamicInferer : public ir::OperationVisitor
{
public:
  DynamicInferer(const ir::Operands &operands, backend::IDynamicTensorManager *tensor_manager,
                 const std::shared_ptr<backend::ITensorRegistry> &tensor_registry)
      : _operands(operands), _dynamic_tensor_manager(tensor_manager),
        _tensor_registry(tensor_registry)
  {
    UNUSED_RELEASE(_operands);
    UNUSED_RELEASE(_dynamic_tensor_manager);
    UNUSED_RELEASE(_tensor_registry);
  }

public:
  // TODO Define visitors for operations. List them in alphabetic order.
  // Remove TODO when any op starting from the alphabet is added
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
  // TODO write op starting from Q
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
  // TODO write op starting from V
  void visit(const ir::operation::ZerosLike &op);

private:
  /**
   * @brief Performs shape inference and memory allocation for arithmetic operation
   */
  void handleBinaryArithmeticOp(const ir::Operation &op, const ir::OperandIndex lhs_idx,
                                const ir::OperandIndex rhs_idx);
  /**
   * @brief Performs shape inference and memory allocation for unary op whose output shape is
   *        always same with input shape
   */
  void handleSimpleUnaryOp(const ir::Operation &op, const ir::OperandIndex input_idx);

private:
  /**
   * @brief To get operand-level info, e.g., ir::Operand::isConstant()
   */
  const ir::Operands &_operands;
  /**
   * @brief To allocate memory for output tensor if needed
   */
  backend::IDynamicTensorManager *_dynamic_tensor_manager;
  /**
   * @brief To get tensor object and access tensor-level info, e.g., ITensor::buffer()
   */
  std::shared_ptr<backend::ITensorRegistry> _tensor_registry;
};

} // namespace shape_inference
} // namespace onert

#endif // __ONERT_GRAPH_SHAPE_INFERENCE_H__
