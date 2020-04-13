/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

/**
 * @file NodeVisitor.h
 * @ingroup COM_AI_RUNTIME
 * @brief This file defines NodeVisitor
 */

#ifndef __INTERNAL_OP_NODE_VISITOR_H__
#define __INTERNAL_OP_NODE_VISITOR_H__

#include "internal/op/Add.h"
#include "internal/op/Sub.h"
#include "internal/op/Mul.h"
#include "internal/op/Div.h"
#include "internal/op/Conv2D.h"
#include "internal/op/DepthwiseConv2D.h"
#include "internal/op/Dequantize.h"
#include "internal/op/MaxPool2D.h"
#include "internal/op/AvgPool2D.h"
#include "internal/op/ArgMax.h"
#include "internal/op/Concat.h"
#include "internal/op/Reshape.h"
#include "internal/op/ResizeBilinear.h"
#include "internal/op/StridedSlice.h"
#include "internal/op/FullyConnected.h"
#include "internal/op/Softmax.h"
#include "internal/op/ReduceMax.h"
#include "internal/op/ReduceMin.h"
#include "internal/op/Cast.h"
#include "internal/op/TopKV2.h"
#include "internal/op/Gather.h"
#include "internal/op/PReLU.h"
#include "internal/op/ReLU.h"
#include "internal/op/ReLU1.h"
#include "internal/op/ReLU6.h"
#include "internal/op/Tanh.h"
#include "internal/op/Squeeze.h"
#include "internal/op/Logistic.h"
#include "internal/op/Mean.h"
#include "internal/op/Rnn.h"
#include "internal/op/Transpose.h"
#include "internal/op/Lstm.h"
#include "internal/op/Floor.h"
#include "internal/op/Split.h"
#include "internal/op/RSQRT.h"
#include "internal/op/SQRT.h"
#include "internal/op/Pad.h"
#include "internal/op/SpaceToDepth.h"
#include "internal/op/SpaceToBatchND.h"
#include "internal/op/L2Pool2D.h"
#include "internal/op/EmbeddingLookup.h"
#include "internal/op/HashtableLookup.h"
#include "internal/op/L2Normalization.h"
#include "internal/op/SquaredDifference.h"
#include "internal/op/LocalResponseNormalization.h"
#include "internal/op/DepthToSpace.h"
#include "internal/op/Unpack.h"
#include "internal/op/Neg.h"
#include "internal/op/Exp.h"
#include "internal/op/ReduceSum.h"
#include "internal/op/Equal.h"
#include "internal/op/BatchToSpaceNd.h"
#include "internal/op/TransposeConv.h"
#include "internal/op/Pack.h"
#include "internal/op/Abs.h"
#include "internal/op/NotEqual.h"
#include "internal/op/LogicalAnd.h"
#include "internal/op/LogicalNot.h"
#include "internal/op/LogicalOr.h"

namespace internal
{
namespace tflite
{
namespace op
{

/**
 * @brief Struct to define visitor for operation Nodes
 */
struct NodeVisitor
{
  /**
   * @brief Destruct NodeVisitor object with default
   */
  virtual ~NodeVisitor() = default;

  /**
   * @brief Visit an Add node
   * @param[in] node Add node to visit
   * @return N/A
   */
  virtual void visit(const Add::Node &) = 0;
  /**
   * @brief Visit a Mul node
   * @param[in] node Mul node to visit
   * @return N/A
   */
  virtual void visit(const Sub::Node &) = 0;
  /**
   * @brief Visit a Mul node
   * @param[in] node Mul node to visit
   * @return N/A
   */
  virtual void visit(const Mul::Node &) = 0;
  /**
   * @brief Visit a Div node
   * @param[in] node Div node to visit
   * @return N/A
   */
  virtual void visit(const Div::Node &) = 0;
  /**
   * @brief Visit a Conv2D node with implicit padding
   * @param[in] node Conv2D node to visit
   * @return N/A
   */
  virtual void visit(const Conv2D::Implicit::Node &) = 0;
  /**
   * @brief Visit a Conv2D node with explicit padding
   * @param[in] node Conv2D node to visit
   * @return N/A
   */
  virtual void visit(const Conv2D::Explicit::Node &) = 0;
  /**
   * @brief Visit a DepthwiseConv2D node with implicit padding
   * @param[in] node DepthwiseConv2D node to visit
   * @return N/A
   */
  virtual void visit(const DepthwiseConv2D::Implicit::Node &) = 0;
  /**
   * @brief Visit a DepthwiseConv2D node with explicit padding
   * @param[in] node DepthwiseConv2D node to visit
   * @return N/A
   */
  virtual void visit(const DepthwiseConv2D::Explicit::Node &) = 0;
  /**
   * @brief Visit a Dequantize node
   * @param[in] node Dequantize node to visit
   * @return N/A
   */
  virtual void visit(const Dequantize::Node &) = 0;
  /**
   * @brief Visit a MaxPool2D node with implicit padding
   * @param[in] node MaxPool2D node to visit
   * @return N/A
   */
  virtual void visit(const MaxPool2D::Implicit::Node &) = 0;
  /**
   * @brief Visit a MaxPool2D node with explicit padding
   * @param[in] node MaxPool2D node to visit
   * @return N/A
   */
  virtual void visit(const MaxPool2D::Explicit::Node &) = 0;
  /**
   * @brief Visit an AvgPool2D node with implicit padding
   * @param[in] node AvgPool2D node to visit
   * @return N/A
   */
  virtual void visit(const AvgPool2D::Implicit::Node &) = 0;
  /**
   * @brief Visit an AvgPool2D node with explicit padding
   * @param[in] node AvgPool2D node to visit
   * @return N/A
   */
  virtual void visit(const AvgPool2D::Explicit::Node &) = 0;
  /**
   * @brief Visit a Concat node
   * @param[in] node Concat node to visit
   * @return N/A
   */
  virtual void visit(const Concat::Node &) = 0;
  /**
   * @brief Visit a ArgMax node
   * @param[in] node ArgMax node to visit
   * @return N/A
   */
  virtual void visit(const ArgMax::Node &) = 0;
  /**
   * @brief Visit an Reshape node
   * @param[in] node Reshape node to visit
   * @return N/A
   */
  virtual void visit(const Reshape::Node &) = 0;
  /**
   * @brief Visit an ResizeBilinear node
   * @param[in] node ResizeBilinear node to visit
   * @return N/A
   */
  virtual void visit(const ResizeBilinear::Node &) = 0;
  /**
   * @brief Visit a StridedSlice node
   * @param[in] node StridedSlice node to visit
   * @return N/A
   */
  virtual void visit(const StridedSlice::Node &) = 0;
  /**
   * @brief Visit a FullyConnected node
   * @param[in] node FullyConnected node to visit
   * @return N/A
   */
  virtual void visit(const FullyConnected::Node &) = 0;
  /**
   * @brief Visit a Softmax node
   * @param[in] node Softmax node to visit
   * @return N/A
   */
  virtual void visit(const Softmax::Node &) = 0;
  /**
   * @brief Visit a ReduceMax node
   * @param[in] node ReduceMax node to visit
   * @return N/A
   */
  virtual void visit(const ReduceMax::Node &) = 0;
  /**
   * @brief Visit a ReduceMin node
   * @param[in] node ReduceMin node to visit
   * @return N/A
   */
  virtual void visit(const ReduceMin::Node &) = 0;
  /**
   * @brief Visit a Cast node
   * @param[in] node Cast node to visit
   * @return N/A
   */
  virtual void visit(const Cast::Node &) = 0;
  /**
   * @brief Visit a TopKV2 node
   * @param[in] node TopKV2 node to visit
   * @return N/A
   */
  virtual void visit(const TopKV2::Node &) = 0;
  /**
   * @brief Visit a Gather node
   * @param[in] node Gather node to visit
   * @return N/A
   */
  virtual void visit(const Gather::Node &) = 0;
  /**
   * @brief Visit an PReLU node
   * @param[in] node PReLU node to visit
   * @return N/A
   */
  virtual void visit(const PReLU::Node &) = 0;
  /**
   * @brief Visit an ReLU node
   * @param[in] node Relu node to visit
   * @return N/A
   */
  virtual void visit(const ReLU::Node &) = 0;
  /**
   * @brief Visit a ReLU1 node
   * @param[in] node ReLU1 node to visit
   * @return N/A
   */
  virtual void visit(const ReLU1::Node &) = 0;
  /**
   * @brief Visit a ReLU6 node
   * @param[in] node ReLU6 node to visit
   * @return N/A
   */
  virtual void visit(const ReLU6::Node &) = 0;
  /**
   * @brief Visit a Tanh node
   * @param[in] node Tanh node to visit
   * @return N/A
   */
  virtual void visit(const Tanh::Node &) = 0;
  /**
   * @brief Visit a Squeeze node
   * @param[in] node Squeeze node to visit
   * @return N/A
   */
  virtual void visit(const Squeeze::Node &) = 0;
  /**
   * @brief Visit an Logistic node
   * @param[in] node Logistic node to visit
   * @return N/A
   */
  virtual void visit(const Logistic::Node &) = 0;
  /**
   * @brief Visit a Mean node
   * @param[in] node Mean node to visit
   * @return N/A
   */
  virtual void visit(const Mean::Node &) = 0;
  /**
   * @brief Visit an RNN node
   * @param[in] node RNN node to visit
   * @return N/A
   */
  virtual void visit(const RNN::Node &) = 0;
  /**
   * @brief Visit a Transpose node
   * @param[in] node Transpose node to visit
   * @return N/A
   */
  virtual void visit(const Transpose::Node &) = 0;
  /**
   * @brief Visit an LSTM node
   * @param[in] node LSTM node to visit
   * @return N/A
   */
  virtual void visit(const LSTM::Node &) = 0;
  /**
   * @brief Visit a Floor node
   * @param[in] node Floor node to visit
   * @return N/A
   */
  virtual void visit(const Floor::Node &) = 0;
  /**
   * @brief Visit a Split node
   * @param[in] node Split node to visit
   * @return N/A
   */
  virtual void visit(const Split::Node &) = 0;
  /**
   * @brief Visit an RSQRT node
   * @param[in] node RSQRT node to visit
   * @return N/A
   */
  virtual void visit(const RSQRT::Node &) = 0;
  /**
   * @brief Visit an SQRT node
   * @param[in] node SQRT node to visit
   * @return N/A
   */
  virtual void visit(const SQRT::Node &) = 0;
  /**
   * @brief Visit a Pad node
   * @param[in] node Pad node to visit
   * @return N/A
   */
  virtual void visit(const Pad::Node &) = 0;
  /**
   * @brief Visit a SpaceToDepth node
   * @param[in] node SpaceToDepth node to visit
   * @return N/A
   */
  virtual void visit(const SpaceToDepth::Node &) = 0;
  /**
   * @brief Visit a SpaceToBatchND node
   * @param[in] node SpaceToBatchND node to visit
   * @return N/A
   */
  virtual void visit(const SpaceToBatchND::Node &) = 0;
  /**
   * @brief Visit an L2Pool2D node with implicit padding
   * @param[in] node L2Pool2D node to visit
   * @return N/A
   */
  virtual void visit(const L2Pool2D::Implicit::Node &) = 0;
  /**
   * @brief Visit an L2Pool2D node with explicit padding
   * @param[in] node L2Pool2D node to visit
   * @return N/A
   */
  virtual void visit(const L2Pool2D::Explicit::Node &) = 0;
  /**
   * @brief Visit an EmbeddingLookup node
   * @param[in] node EmbeddingLookup node to visit
   * @return N/A
   */
  virtual void visit(const EmbeddingLookup::Node &) = 0;
  /**
   * @brief Visit a HashtableLookup node
   * @param[in] node HashtableLookup node to visit
   * @return N/A
   */
  virtual void visit(const HashtableLookup::Node &) = 0;
  /**
   * @brief Visit an L2Normalization node
   * @param[in] node L2Normalization node to visit
   * @return N/A
   */
  virtual void visit(const L2Normalization::Node &) = 0;
  /**
   * @brief Visit a SquaredDifference node
   * @param[in] node SquaredDifference node to visit
   * @return N/A
   */
  virtual void visit(const SquaredDifference::Node &) = 0;
  /**
   * @brief Visit a LocalResponseNormalization node
   * @param[in] node LocalResponseNormalization node to visit
   * @return N/A
   */
  virtual void visit(const LocalResponseNormalization::Node &) = 0;
  /**
   * @brief Visit a DepthToSpace node
   * @param[in] node DepthToSpace node to visit
   * @return N/A
   */
  virtual void visit(const DepthToSpace::Node &) = 0;
  /**
   * @brief Visit a Unpack node
   * @param[in] node Unpack node to visit
   * @return N/A
   */
  virtual void visit(const Unpack::Node &) = 0;
  /**
   * @brief Visit a Neg node
   * @param[in] node Neg node to visit
   * @return N/A
   */
  virtual void visit(const Neg::Node &) = 0;
  /**
   * @brief Visit a Exp node
   * @param[in] node Exp node to visit
   * @return N/A
   */
  virtual void visit(const Exp::Node &) = 0;
  /**
   * @brief Visit a ReduceSum node
   * @param[in] node ReduceSum node to visit
   * @return N/A
   */
  virtual void visit(const ReduceSum::Node &) = 0;
  /**
   * @brief Visit a Equal node
   * @param[in] node Equal node to visit
   * @return N/A
   */
  virtual void visit(const Equal::Node &) = 0;
  /**
   * @brief Visit a BatchToSpaceNd node
   * @param[in] node BatchToSpaceNd node to visit
   * @return N/A
   */
  virtual void visit(const BatchToSpaceNd::Node &) = 0;
  /**
   * @brief Visit a TransposeConv node
   * @param[in] node TransposeConv node to visit
   * @return N/A
   */
  virtual void visit(const TransposeConv::Node &) = 0;
  /**
   * @brief Visit a Pack node
   * @param[in] node Pack node to visit
   * @return N/A
   */
  virtual void visit(const Pack::Node &) = 0;
  /**
   * @brief Visit a Abs node
   * @param[in] node Abs node to visit
   * @return N/A
   */
  virtual void visit(const Abs::Node &) = 0;
  /**
   * @brief Visit a NotEqual node
   * @param[in] node NotEqual node to visit
   * @return N/A
   */
  virtual void visit(const NotEqual::Node &) = 0;
  /**
   * @brief Visit a LogicalAnd node
   * @param[in] node LogicalAnd node to visit
   * @return N/A
   */
  virtual void visit(const LogicalAnd::Node &) = 0;
  /**
   * @brief Visit a LogicalNot node
   * @param[in] node LogicalNot node to visit
   * @return N/A
   */
  virtual void visit(const LogicalNot::Node &) = 0;
  /**
   * @brief Visit a LogicalOr node
   * @param[in] node LogicalOr node to visit
   * @return N/A
   */
  virtual void visit(const LogicalOr::Node &) = 0;
};

} // namespace op
} // namespace tflite
} // namespace internal

#endif // __INTERNAL_OP_NODE_VISITOR_H__
