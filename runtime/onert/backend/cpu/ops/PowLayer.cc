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

#include "PowLayer.h"

#include "../KernelGenerator.h"
#include "../Validator.h"

#include <cker/operation/Pow.h>
#include <cker/operation/BinaryArithmeticOps.h>

namespace onert::backend::cpu
{

void Validator::visit(const ir::operation::Pow &) { _supported = true; }

void KernelGenerator::visit(const ir::operation::Pow &node)
{
  const auto output_index{node.getOutputs().at(0)};
  const auto lhs_index{node.getInputs().at(ir::operation::Pow::LHS)};
  const auto rhs_index{node.getInputs().at(ir::operation::Pow::RHS)};

  auto output_tensor = _tensor_reg->getPortableTensor(output_index);
  auto lhs_tensor = _tensor_reg->getPortableTensor(lhs_index);
  auto rhs_tensor = _tensor_reg->getPortableTensor(rhs_index);

  auto fn = std::make_unique<ops::PowLayer>();

  fn->configure(lhs_tensor, rhs_tensor, ir::Activation::NONE, output_tensor);

  _return_fn = std::move(fn);
}

} // namespace onert::backend::cpu

namespace onert::backend::cpu::ops
{

void PowLayer::powFloat32()
{
  float output_activation_min = 0, output_activation_max = 0;
  CalculateActivationRange(_activation, &output_activation_min, &output_activation_max);
  nnfw::cker::BinaryArithmeticOpParam op_params;
  op_params.float_activation_max = output_activation_max;
  op_params.float_activation_min = output_activation_min;

  if (!HaveSameShapes(_lhs, _rhs))
  {
    nnfw::cker::BroadcastBinaryArithmeticOp<nnfw::cker::BinaryArithmeticOpType::POW>(
      op_params, getShape(_lhs), getBuffer<float>(_lhs), getShape(_rhs), getBuffer<float>(_rhs),
      getShape(_output), getBuffer<float>(_output));
    return;
  }

  nnfw::cker::powImpl(getShape(_lhs), getBuffer<float>(_lhs), getShape(_rhs),
                      getBuffer<float>(_rhs), getShape(_output), getBuffer<float>(_output));
}

void PowLayer::configure(const IPortableTensor *lhs, const IPortableTensor *rhs,
                         ir::Activation activation, IPortableTensor *output)
{
  _lhs = lhs;
  _rhs = rhs;
  _activation = activation;
  _output = output;
}

void PowLayer::run()
{
  if (_output->data_type() == OperandType::FLOAT32)
    powFloat32();
  else
    throw std::runtime_error{"Pow: unsupportted data type"};
}

} // namespace onert::backend::cpu::ops
