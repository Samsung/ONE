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

void ShapeFixer::visit(const ir::operation::Pack &node)
{
  const auto ofm_index{node.getOutputs().at(0)};
  for (const auto &inputs : node.getInputs())
  {
    const auto ofm_rank = _ctx.at(ofm_index).shape().rank();

    // TODO remove const_cast later. For example, _ctx may need to be a non const variable or
    //      a node to extend shape may be inserted in front of this operation
    const_cast<ir::Shape &>(_ctx.at(inputs).shape()).extendRank(ofm_rank);
  }
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

} // namespace acl_neon
} // namespace backend
} // namespace onert
