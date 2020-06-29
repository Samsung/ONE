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

#include "ops/AddLayer.h"
#include "ops/AvgPoolLayer.h"
#include "ops/CastLayer.h"
#include "ops/ConcatLayer.h"
#include "ops/ConvolutionLayer.h"
#include "ops/DepthwiseConvolutionLayer.h"
#include "ops/DivLayer.h"
#include "ops/ExpLayer.h"
#include "ops/FullyConnectedLayer.h"
#include "ops/GatherLayer.h"
#include "ops/MaxPoolLayer.h"
#include "ops/MulLayer.h"
#include "ops/OperationUtils.h"
#include "ops/ReshapeLayer.h"
#include "ops/SoftMaxLayer.h"
#include "ops/SubLayer.h"

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

void ShapeFixer::visit(const ir::operation::Add &) { /* DO NOTHING */}

void ShapeFixer::visit(const ir::operation::Sub &node)
{
  // The same as Add
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Sub"};
  }
}

void ShapeFixer::visit(const ir::operation::Div &node)
{
  // The same as Add
  const auto lhs_index{node.getInputs().at(ir::operation::Div::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Div"};
  }
}

void ShapeFixer::visit(const ir::operation::Pad &node)
{
  // TODO: empty this method when quantization is supported
  const auto lhs_index{node.getInputs().at(ir::operation::Sub::Input::LHS)};

  // Quantization : not supported
  if (_ctx.at(lhs_index).typeInfo().type() == ir::DataType::QUANT_UINT8_ASYMM)
  {
    throw std::runtime_error{"ShapeFixer: NYI for quantized Pad"};
  }
}

} // namespace cpu
} // namespace backend
} // namespace onert
