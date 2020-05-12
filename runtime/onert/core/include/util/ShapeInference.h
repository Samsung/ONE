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

#include "ir/operation/AvgPool2D.h"
#include "ir/operation/Concat.h"
#include "ir/operation/MaxPool2D.h"
#include "ir/operation/Conv2D.h"
#include "ir/operation/DepthwiseConv2D.h"
#include "ir/operation/Reshape.h"
#include "ir/Operands.h"
#include "ir/Index.h"
#include "ir/Layout.h"
#include "ir/OperationVisitor.h"

namespace onert
{
namespace shape_inference
{

using Shapes = std::vector<ir::Shape>;

// Define shape calculation for operations. List them in alphabetic order.
// Remove TODO when the function name matching the alphabet is added

Shapes inferAvgPoolShape(const ir::Shape &in_shape, const ir::operation::AvgPool2D::Param &param,
                         ir::Layout layout = ir::Layout::NHWC);

ir::Shape inferConcatShape(const Shapes &in_shapes, const ir::operation::Concat::Param &param);

Shapes inferConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                        const ir::operation::Conv2D::Param &param,
                        ir::Layout layout = ir::Layout::NHWC);

Shapes inferDepthwiseConv2DShape(const ir::Shape &in_shape, const ir::Shape &ker_shape,
                                 const ir::operation::DepthwiseConv2D::Param &param,
                                 ir::Layout layout = ir::Layout::NHWC);

Shapes inferEltwiseShape(const ir::Shape &lhs_shape, const ir::Shape &rhs_shape);

Shapes inferFullyConnectedShape(const ir::Shape &in_shape, const ir::Shape &ker_shape);

// TODO write op starting from G
// TODO write op starting from L

Shapes inferMaxPoolShape(const ir::Shape &in_shape, const ir::operation::MaxPool2D::Param &param,
                         ir::Layout layout = ir::Layout::NHWC);

// TODO write op starting from N
// TODO write op starting from P
// TODO write op starting from R
// TODO write op starting from S
// TODO write op starting from T
// TODO write op starting from U
// TODO write op starting from Z

/**
 * @brief Class to infer shape before running kernels. It does the following:
 *        - re-calculate and set output shape at compile time (before running kernels)
 *        - if calculation cannot be done at compile time, mark the outputs to be dynamic, meaning
 *          shapes of outputs will be calculated during running kernels
 */
class StaticInferer : public ir::OperationVisitor
{
public:
  StaticInferer(ir::Operands &operands) : _operands(operands) { /* empty */}
  virtual ~StaticInferer() = default;

public:
  /**
   * @brief Infer shape of operands beloning to ops and set the output shape.
   *        If output shape cannot be known without running op, mark it so that it can be allocated
   *        when running kernel.
   * @param op_seq sequence of operations
   */
  void infer(const ir::OpSequence &op_seq) { op_seq.accept(*this); };

private:
  // TODO Define visitors for operations. List them in alphabetic order.
  // Remove TODO when any op starting from the alphabet is added
  void visit(const ir::operation::Add &op);
  void visit(const ir::operation::Concat &op);
  // TODO write op starting from D
  // TODO write op starting from E
  // TODO write op starting from F
  // TODO write op starting from G
  // TODO write op starting from L
  // TODO write op starting from M
  // TODO write op starting from N
  // TODO write op starting from P
  void visit(const ir::operation::Reshape &op);
  // TODO write op starting from S
  // TODO write op starting from T
  // TODO write op starting from U
  // TODO write op starting from Z

private:
  ir::Operands &_operands;
};

} // namespace shape_inference
} // namespace onert

#endif // __ONERT_GRAPH_SHAPE_INFERENCE_H__
