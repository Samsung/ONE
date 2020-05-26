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

#include "ShapeFixer.h"

#include "kernel/AddLayer.h"
#include "kernel/AvgPoolLayer.h"
#include "kernel/CastLayer.h"
#include "kernel/ConcatLayer.h"
#include "kernel/ConvolutionLayer.h"
#include "kernel/DepthwiseConvolutionLayer.h"
#include "kernel/DivLayer.h"
#include "kernel/ExpLayer.h"
#include "kernel/FullyConnectedLayer.h"
#include "kernel/GatherLayer.h"
#include "kernel/MaxPoolLayer.h"
#include "kernel/MulLayer.h"
#include "kernel/OperationUtils.h"
#include "kernel/ReshapeLayer.h"
#include "kernel/SoftMaxLayer.h"
#include "kernel/SubLayer.h"

#include <backend/Backend.h>
#include <backend/IConfig.h>
#include <memory>
#include <util/Utils.h>
#include <util/logging.h>

#include <stdexcept>

namespace onert
{
namespace backend
{
namespace cpu
{

ShapeFixer::ShapeFixer(const ir::Operands &operand_ctx) : _ctx(operand_ctx) {}

void ShapeFixer::visit(const ir::operation::Add &node)
{
  const auto lhs_index{node.getInputs().at(ir::operation::Add::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Add"};
  }
}

void ShapeFixer::visit(const ir::operation::Sub &node)
{
  // The same as Add
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Sub"};
  }
}

void ShapeFixer::visit(const ir::operation::Mul &node)
{
  // The same as Add
  const auto lhs_index{node.getInputs().at(ir::operation::Mul::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Mul"};
  }
}

void ShapeFixer::visit(const ir::operation::Div &node)
{
  // The same as Add
  const auto lhs_index{node.getInputs().at(ir::operation::Div::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Div"};
  }
}

void ShapeFixer::visit(const ir::operation::Pad &node)
{
  // TODO: empty this method when quantization is supported
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Pad"};
  }
}

} // namespace cpu
} // namespace backend
} // namespace onert
