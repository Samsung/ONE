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

#include <cker/operation/Pow.h>
#include <cker/operation/BinaryArithmeticOps.h>

namespace onert
{
namespace backend
{
namespace training
{
namespace ops
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

} // namespace ops
} // namespace training
} // namespace backend
} // namespace onert
