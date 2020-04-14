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

#include "ShapeFixer.h"

#include <arm_compute/runtime/NEON/functions/NESoftmaxLayer.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticAddition.h>
#include <arm_compute/runtime/NEON/functions/NEArithmeticSubtraction.h>
#include <arm_compute/runtime/NEON/functions/NEPixelWiseMultiplication.h>
#include <arm_compute/runtime/NEON/functions/NEPoolingLayer.h>
#include <arm_compute/runtime/NEON/functions/NEActivationLayer.h>
#include <arm_compute/runtime/NEON/functions/NEConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEDepthwiseConvolutionLayer.h>
#include <arm_compute/runtime/NEON/functions/NEReshapeLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedLayer.h>
#include <arm_compute/runtime/NEON/functions/NEFullyConnectedReshapingLayer.h>

#include <Convert.h>
#include <Swizzle.h>

#include "ir/Index.h"
#include "exec/NopFunction.h"
#include "util/logging.h"
#include "util/Utils.h"

namespace onert
{
namespace backend
{
namespace acl_neon
{

using ::onert::backend::acl_common::asAclFunction;

ShapeFixer::ShapeFixer(const ir::Operands &ctx,
                       const std::shared_ptr<TensorBuilder> &tensor_builder)
    : _ctx(ctx), _tensor_builder(tensor_builder)
{
  assert(tensor_builder);
}

void ShapeFixer::visit(const ir::operation::Abs &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ArgMax &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::BatchToSpaceND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::BatchToSpaceND::Input::INPUT)};
  _tensor_builder->dimCorrection(ofm_index, false);
  _tensor_builder->dimCorrection(ifm_index, false);
}

void ShapeFixer::visit(const ir::operation::Cast &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Conv2D &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::DepthToSpace &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::DepthwiseConv2D &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Dequantize &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::MaxPool2D &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Mean &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::AvgPool2D &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Concat &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  _tensor_builder->dimCorrection(ofm_index, false);
  for (const auto &inputs : node.getInputs())
    _tensor_builder->dimCorrection(inputs, false);
}

void ShapeFixer::visit(const ir::operation::EmbeddingLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};
  _tensor_builder->dimCorrection(values_index, false);
  _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::Exp &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Floor &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::FullyConnected &node)
{
  using ir::operation::FullyConnected;
  const auto input_index{node.getInputs().at(FullyConnected::Input::INPUT)};
  const auto input_rank = _ctx.at(input_index).shape().rank();
  // Check for reshaping input's shape into rank-2
  if (input_rank == 3 || input_rank == 4)
    _tensor_builder->dimCorrection(input_index, false);
}

void ShapeFixer::visit(const ir::operation::HashtableLookup &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto values_index{node.getInputs().at(ir::operation::EmbeddingLookup::Input::VALUES)};
  _tensor_builder->dimCorrection(values_index, false);
  _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::Gather &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::Gather::Input::INPUT)};
  const auto indices_index{node.getInputs().at(ir::operation::Gather::Input::INDICES)};
  _tensor_builder->dimCorrection(ofm_index, false);
  _tensor_builder->dimCorrection(ifm_index, false);
  _tensor_builder->dimCorrection(indices_index, false);
}

void ShapeFixer::visit(const ir::operation::InstanceNorm &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::L2Normalization &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::L2Pool2D &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::LocalResponseNormalization &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::LogicalAnd &node)
{
  const auto input0_index{node.getInputs().at(ir::operation::LogicalAnd::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::LogicalAnd::Input::INPUT1)};

  if (!(_ctx.at(input0_index).shape() == _ctx.at(input1_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input0_index).shape().rank(), _ctx.at(input1_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(input0_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(input1_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::LogicalNot &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::LogicalOr &node)
{
  const auto input0_index{node.getInputs().at(ir::operation::LogicalOr::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::LogicalOr::Input::INPUT1)};

  if (!(_ctx.at(input0_index).shape() == _ctx.at(input1_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input0_index).shape().rank(), _ctx.at(input1_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(input0_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(input1_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Logistic &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::LSTM &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  _tensor_builder->dimCorrection(ofm_index, false);
  for (const auto &inputs : node.getInputs())
  {
    _tensor_builder->dimCorrection(inputs, false);
    const auto ofm_rank = _ctx.at(ofm_index).shape().rank();

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(inputs).shape()).extendRank(ofm_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Pad &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Pad::Input::INPUT)};
  const auto output_index{node.getOutputs().at(0)};
  _tensor_builder->dimCorrection(input_index, false);
  _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::Mul &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Mul::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Mul::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Neg &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Permute &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::PReLU &node)
{
  const auto ifm_index{node.getInputs().at(ir::operation::PReLU::Input::INPUT)};
  const auto alpha_index{node.getInputs().at(ir::operation::PReLU::Input::ALPHA)};

  if (!(_ctx.at(ifm_index).shape() == _ctx.at(alpha_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(ifm_index).shape().rank(), _ctx.at(alpha_index).shape().rank());
    const_cast<ir::Shape &>(_ctx.at(ifm_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(alpha_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::ReduceMax &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ReduceMin &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ReduceSum &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ReLU &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ReLU1 &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::ReLU6 &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Reshape &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Reshape::Input::INPUT)};

  _tensor_builder->dimCorrection(input_index, false);
  _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::ResizeBilinear &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::RNN &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Comparison &node)
{
  const auto input0_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT0)};
  const auto input1_index{node.getInputs().at(ir::operation::Comparison::Input::INPUT1)};

  if (!(_ctx.at(input0_index).shape() == _ctx.at(input1_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(input0_index).shape().rank(), _ctx.at(input1_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(input0_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(input1_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::RSQRT &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Squeeze &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto input_index{node.getInputs().at(ir::operation::Squeeze::Input::INPUT)};
  _tensor_builder->dimCorrection(input_index, false);
  _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::Tanh &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Slice &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::StridedSlice &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::StridedSlice::Input::INPUT)};
  _tensor_builder->dimCorrection(ofm_index, false);
  _tensor_builder->dimCorrection(ifm_index, false);
}

void ShapeFixer::visit(const ir::operation::Softmax &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::SpaceToBatchND &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  const auto ifm_index{node.getInputs().at(ir::operation::SpaceToBatchND::Input::INPUT)};
  _tensor_builder->dimCorrection(ofm_index, false);
  _tensor_builder->dimCorrection(ifm_index, false);
}

void ShapeFixer::visit(const ir::operation::SpaceToDepth &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Split &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Split::Input::INPUT)};
  _tensor_builder->dimCorrection(input_index, false);
  for (const auto &output : node.getOutputs())
    _tensor_builder->dimCorrection(output, false);
}

void ShapeFixer::visit(const ir::operation::SQRT &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::SquaredDifference &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::SquaredDifference::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Sub &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Sub::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::TransposeConv &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Transpose &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Unpack &node)
{
  const auto input_index{node.getInputs().at(ir::operation::Unpack::Input::INPUT)};
  _tensor_builder->dimCorrection(input_index, false);
  for (const auto &output_index : node.getOutputs())
    _tensor_builder->dimCorrection(output_index, false);
}

void ShapeFixer::visit(const ir::operation::Add &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Add::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Add::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Div &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Div::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Div::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Min &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Min::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Min::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::Max &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Max::Input::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Max::Input::RHS)};

  if (!(_ctx.at(lhs_index).shape() == _ctx.at(rhs_index).shape()))
  {
    const auto broadcast_rank =
        std::max(_ctx.at(lhs_index).shape().rank(), _ctx.at(rhs_index).shape().rank());

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(lhs_index).shape()).extendRank(broadcast_rank);
    const_cast<ir::Shape &>(_ctx.at(rhs_index).shape()).extendRank(broadcast_rank);
  }
}

void ShapeFixer::visit(const ir::operation::OneHot &) { /* Do NOTHING */}

} // namespace acl_neon
} // namespace backend
} // namespace onert
